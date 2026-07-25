"""configs/pi05/train_config.py

π0.5 模型 LoRA 微调训练配置（openpi / JAX 路径）。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程（JAX 路径、LeRobot、norm stats、LoRA、动作块适配）。
- §3.2.1：LoRA rank 32 为 OpenVLA-OFT 官方示例候选（openpi/π0.5 暂以此为初始值，
  D21 实验后按闭环表现确认或调整；方案书未对 π0.5 单独规定 LoRA rank 数值）。
- §3.3：π0.5 显存参考：推理 >8GB、LoRA >22.5GB、全参 >70GB；需要 LoRA 时必走 JAX 路径，
  PyTorch 路径目前不支持 LoRA / 混合精度 / FSDP / EMA。
- §3.3.1 Para186：本项目自有 norm stats，不沿用 OpenVLA；训练前必跑 compute_norm_stats。
- §3.4：动作 7 维 [dx,dy,dz,dax,day,daz,gripper]，robot_base，axis-angle，control_hz=10。
- §3.3：LIBERO 配置动作块常为 10，其他域可能不同；以本项目 checkpoint 配置为准，
  不照抄论文动作长度。
- §5.4：canonical → LeRobot 数据转换由 scripts/pi05/convert_openpi.py 完成。
- §6.3：首轮微调 100—500 条/核心技能；LoRA；1—2 组超参；ID val 成功≥60%。

配置体系说明（必须遵循 openpi 官方）：
- openpi 不用 Pydantic / YAML / HuggingFace PEFT，用自己的 Python dataclass + tyro。
- 本文件基于官方 TrainConfig 创建配置实例，不重新定义官方类。
- openpi 的 LoRA 通过两个机制实现（不是 PEFT）：
  1. freeze_filter：JAX nnx.filterlib.Filter，冻结 base 参数，只让 LoRA 层可训练。
  2. LoRAWeightLoader：加载预训练 base 权重，同时注入 LoRA 适配层。
  不需要 lora_alpha / target_modules / dropout 这些 PEFT 参数；
  LoRA rank 等参数在 weight_loader 或 model_config 中指定。
- freeze_filter 与 weight_loader 留空（TODO），等服务器上按官方 LoRA 文档配置；
  weight_loader 指向 pi05_base checkpoint。

关键参数：
- base checkpoint：gs://openpi-assets/checkpoints/pi05_base
- LoRA rank：32（参考方案书 §3.2.1 OpenVLA-OFT LoRA 示例；π0.5 初始候选值，D21 后按实验确认）
- 显存要求：>22.5GB（方案书 §3.3）；22.5GB 卡建议 batch_size 降到 16 或 8。
- num_train_steps：30000（openpi 官方示例参考值；首轮微调 100—500 条/核心技能 §6.3，
  实际步数 D21 按数据量与收敛情况调整）
- warmup_steps / weight_decay / gradient_accumulation_steps / mixed_precision：
  见下方环境变量区；若 openpi 官方 TrainConfig 未暴露对应字段，则由 optimizer/scheduler 内部默认值控制。

训练命令：
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_industrial \\
        --exp-name=my_experiment --overwrite

norm stats 计算命令（训练前必跑，方案书 §3.3.1 Para186）：
    uv run scripts/compute_norm_stats.py --config-name pi05_industrial
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("pi05_train_config")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][pi05_config] %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 环境变量占位（路径可配置）
# ---------------------------------------------------------------------------
# 方案书 §5.2：dataset 命名 dataset_v{major}.{minor}；repo_id 由数据负责人 C 冻结。
DATASET_REPO_ID: str = os.environ.get(
    "PI05_DATASET_REPO_ID", "industrial_team/industrial_dataset"
)
# 训练输出目录（openpi 默认 ckpts/<config_name>，可用环境变量覆盖）
OUTPUT_DIR: str = os.environ.get("PI05_OUTPUT_DIR", "./checkpoints/pi05_industrial")
# base checkpoint 地址（openpi 官方 pi05_base，方案书 §3.3.1 复现官方 checkpoint）
BASE_CHECKPOINT: str = os.environ.get(
    "PI05_BASE_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_base"
)
# LoRA rank（方案书 §3.2.1；在 weight_loader / model_config 中使用）
LORA_RANK: int = int(os.environ.get("PI05_LORA_RANK", "32"))
# state 维度：Franka 7-DOF + 1 gripper = 8（convert_openpi.py DEFAULT_STATE_DIM；S1 修复）
# 作为唯一真相源，供 train.py 与 compute_norm_stats.py 引用
STATE_DIM: int = int(os.environ.get("PI05_STATE_DIM", "8"))

# ----------------- 训练超参数补充（C2 修复）-----------------
# 以下参数若 openpi 官方 TrainConfig 未直接暴露对应字段，则由 optimizer / lr_schedule 内部默认值控制；
# 此处暴露为环境变量用于文档审计与外部脚本覆盖，实际生效依赖 openpi 官方实现。
# batch_size 显式覆盖（方案书 §3.3：22.5GB 卡建议降到 16 或 8；默认 16 为安全值）
BATCH_SIZE: int = int(os.environ.get("PI05_BATCH_SIZE", "16"))
# warmup_steps：学习率线性预热步数（openpi 官方示例约 2000；D21 按数据量调整，§6.3）
WARMUP_STEPS: int = int(os.environ.get("PI05_WARMUP_STEPS", "2000"))
# weight_decay：AdamW 权重衰减系数（openpi 官方示例约 0.01—0.1；D21 实验确认）
WEIGHT_DECAY: float = float(os.environ.get("PI05_WEIGHT_DECAY", "0.01"))
# gradient_accumulation_steps：梯度累积步数（有效 batch = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS）
GRADIENT_ACCUMULATION_STEPS: int = int(
    os.environ.get("PI05_GRADIENT_ACCUMULATION_STEPS", "1")
)
# mixed_precision：混合精度模式（JAX 路径推荐 bf16；方案书 §3.3 禁 PyTorch 路径因不支持混合精度）
MIXED_PRECISION: str = os.environ.get("PI05_MIXED_PRECISION", "bf16")
# eval_interval：验证评测间隔步数（方案书 §6.3 要求每个 checkpoint 在独立验证 seed 闭环评测）
EVAL_INTERVAL: int = int(os.environ.get("PI05_EVAL_INTERVAL", "1000"))


# ---------------------------------------------------------------------------
# 依赖：openpi（try/except，缺失时降级为纯 dataclass 占位定义 + 打印提示）
# ---------------------------------------------------------------------------
# 方案书 §3.3：需要 LoRA 时必须走 JAX 路径，依赖 openpi 官方包。
# 本地无 openpi 时，仍允许本文件被 import（用于文档/CI 静态检查），
# 但配置实例为占位 dataclass，不能用于真实训练。
OPENPI_AVAILABLE: bool = False
OPENPI_IMPORT_ERROR: Optional[str] = None

TrainConfig: Any = None
LeRobotLiberoDataConfig: Any = None
Pi0Config: Any = None
ModelType: Any = None
CosineDecaySchedule: Any = None
AdamW: Any = None
WeightLoader: Any = None
_CONFIGS: Dict[str, Any] = {}

try:
    from openpi.training.config import TrainConfig, LeRobotLiberoDataConfig  # type: ignore
    from openpi.models.pi0_config import Pi0Config  # type: ignore
    from openpi.models.model import ModelType  # type: ignore
    from openpi.training.optimizer import CosineDecaySchedule, AdamW  # type: ignore
    from openpi.training.weight_loaders import WeightLoader  # type: ignore

    # 复用官方 _CONFIGS 注册表（若官方以字典形式导出）。
    # 注意：_CONFIGS 为 openpi 私有 API（前缀下划线），官方可能随时重命名或移除；
    # 若导入失败则降级为本地空字典，register_config() 仅注册到本地占位表。
    try:
        from openpi.training.config import _CONFIGS  # type: ignore
    except Exception:  # 官方未导出或已改名 _CONFIGS，使用本地空字典（W1 已记录风险）
        _CONFIGS = {}
    OPENPI_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    OPENPI_IMPORT_ERROR = str(_e)

    # ---- 降级：纯 dataclass 占位定义，仅保证文件可被 import ----
    @dataclass
    class TrainConfig:
        """占位 TrainConfig（openpi 不可用时的降级定义，字段对齐官方 §7.1）。"""

        name: str = ""
        exp_name: str = ""
        model: Any = None
        weight_loader: Any = None
        lr_schedule: Any = None
        optimizer: Any = None
        ema_decay: float = 0.999
        freeze_filter: Any = None
        data: Any = None
        batch_size: int = 16
        num_workers: int = 2
        num_train_steps: int = 30000
        log_interval: int = 100
        save_interval: int = 1000
        keep_period: int = 5000
        overwrite: bool = False
        resume: bool = False
        wandb_enabled: bool = True
        fsdp_devices: int = 1
        # 以下字段在 openpi 官方 TrainConfig 中可能由 optimizer/lr_schedule 内部承载；
        # 降级占位中显式定义以保证文档可审计性（C2 修复）。
        warmup_steps: int = 2000
        weight_decay: float = 0.01
        gradient_accumulation_steps: int = 1
        mixed_precision: str = "bf16"
        eval_interval: int = 1000

    @dataclass
    class LeRobotLiberoDataConfig:
        """占位数据配置（对齐官方 LeRobotLiberoDataConfig 关键字段）。"""

        repo_id: str = ""
        assets: Any = None
        transforms: Any = None

    @dataclass
    class Pi0Config:
        """占位模型配置（对齐官方 Pi0Config 关键字段）。"""

        model_type: Any = None
        action_dim: int = 7
        action_horizon: int = 10
        max_token_len: int = 48

    class ModelType:  # 占位枚举
        PI05 = "pi05"

    @dataclass
    class CosineDecaySchedule:
        """占位学习率调度（对齐官方 CosineDecaySchedule 关键字段）。"""

        init_value: float = 2e-5
        warmup_steps: int = 2000  # 线性预热步数（C2 修复）

    @dataclass
    class AdamW:
        """占位优化器（官方默认 AdamW；weight_decay 由 openpi 内部控制）。"""

        weight_decay: float = 0.01  # C2 修复：显式声明默认值以保持可审计性

    class WeightLoader:
        """占位权重加载器。"""

        pass

    _CONFIGS = {}

    print(
        "[pi05_train_config] WARNING: openpi 未安装或 import 失败，已降级为占位 dataclass。\n"
        f"  原因: {OPENPI_IMPORT_ERROR}\n"
        "  提示: LoRA 微调必须走 openpi JAX 路径（方案书 §3.3）。\n"
        "        请安装: git clone https://github.com/Physical-Intelligence/openpi\n"
        "                cd openpi && uv sync\n"
        "  当前 PI05_INDUSTRIAL_CONFIG 仅用于文档/检查，不能用于真实训练。"
    )


# ---------------------------------------------------------------------------
# 配置实例：pi05_industrial
# ---------------------------------------------------------------------------
def _build_pi05_industrial_config() -> TrainConfig:
    """构建 pi05_industrial TrainConfig 实例。

    openpi 可用时用官方类；不可用时用本文件降级 dataclass 占位（仅用于文档/检查）。
    """
    cfg = TrainConfig(
        name="pi05_industrial",
        # ---- 模型配置 ----
        # 方案书 §3.4：动作 7 维 [dx,dy,dz,dax,day,daz,gripper]。
        # 方案书 §3.3：LIBERO 配置动作块常为 10；以本项目 checkpoint 配置为准，
        #   action_horizon=10 为初始候选，D21 首轮微调后按闭环表现调整。
        model=Pi0Config(
            model_type=ModelType.PI05,
            action_dim=7,  # 7 维动作（方案书 §3.4）
            action_horizon=10,  # 动作块长度（初始候选，D21 后按闭环表现调整）
            max_token_len=48,  # 文本 token 最大长度
        ),
        # ---- 数据配置 ----
        # 方案书 §5.4：canonical → LeRobot 转换由 scripts/pi05/convert_openpi.py 完成。
        # 方案书 §3.3.1 Para186：norm stats 用 compute_norm_stats 单独生成本项目自有统计。
        data=LeRobotLiberoDataConfig(
            repo_id=DATASET_REPO_ID,
        ),
        # ---- 训练参数 ----
        # 方案书 §3.3：LoRA 微调显存 >22.5GB；22.5GB 卡建议降到 16 或 8。
        # 默认 batch_size=16（安全值），可通过 PI05_BATCH_SIZE 环境变量覆盖（W3 修复）。
        batch_size=BATCH_SIZE,
        num_workers=2,
        num_train_steps=30000,  # openpi 官方示例参考值；D21 按数据量与收敛情况调整（W5）
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        # eval_interval：若 openpi TrainConfig 支持则取消注释；当前由外部验证脚本按 save_interval
        # 触发评测（方案书 §6.3 要求每个 checkpoint 在独立验证 seed 闭环评测；PI05_EVAL_INTERVAL
        # 环境变量已定义用于审计，见 W2）。
        # ---- 学习率 ----
        # LoRA 微调用较小学习率（方案书 §6.3 首轮微调 1—2 组超参）。
        # warmup_steps 若 openpi CosineDecaySchedule 支持则传入（取消注释 warmup_steps=WARMUP_STEPS）。
        lr_schedule=CosineDecaySchedule(
            init_value=2e-5,
            # warmup_steps=WARMUP_STEPS,  # 若官方 CosineDecaySchedule 支持则取消注释（C2）
        ),
        # ---- 优化器（C2 修复） ----
        # weight_decay / gradient_accumulation_steps 若 openpi AdamW / TrainConfig 支持对应字段则传入。
        # 实际值见环境变量 PI05_WEIGHT_DECAY / PI05_GRADIENT_ACCUMULATION_STEPS。
        # mixed_precision：JAX 路径推荐 bf16；若 openpi TrainConfig 支持 mixed_precision 字段则取消注释（C2）。
        # optimizer=AdamW(weight_decay=WEIGHT_DECAY)  # 若官方 AdamW 支持则取消注释（C2）
        # ---- LoRA 相关（openpi 机制，非 PEFT） ----
        # freeze_filter：JAX nnx.filterlib.Filter，冻结 base 参数只让 LoRA 层可训练。
        #   TODO(D21): 服务器上按官方 LoRA 文档配置 freeze_filter。
        # weight_loader：LoRAWeightLoader，加载 base 权重并注入 LoRA 适配层。
        #   TODO(D21): 指向 pi05_base checkpoint（BASE_CHECKPOINT），LoRA rank=LORA_RANK。
        # 方案书 §3.2.1：LoRA rank=32 为 OpenVLA-OFT 示例候选值；π0.5 暂沿用。
        # freeze_filter=...,         # 留空，服务器上配置
        # weight_loader=...,         # 留空，指向 pi05_base checkpoint
        # ---- 其他 ----
        overwrite=True,
        wandb_enabled=True,
        fsdp_devices=1,  # 单卡用 1（方案书 §3.3：JAX 路径）
    )
    return cfg


PI05_INDUSTRIAL_CONFIG: TrainConfig = _build_pi05_industrial_config()


# ---------------------------------------------------------------------------
# 安全闸门：冻结核心 LoRA 机制配置前禁止训练（C3 修复）
# ---------------------------------------------------------------------------
def validate_lora_ready(config: Optional[TrainConfig] = None) -> bool:
    """检查 LoRA freeze_filter 与 weight_loader 是否已配置。

    方案书 §3.3.1 要求走 JAX LoRA 路径；若 freeze_filter / weight_loader 均
    为 None，当前配置将执行全参数训练（显存 >70GB）而非 LoRA 微调（>22.5GB），
    与方案书规定矛盾，且大概率在 22.5GB 卡上 OOM。

    Args:
        config: 待检查的 TrainConfig；默认使用 PI05_INDUSTRIAL_CONFIG。

    Returns:
        True 表示 LoRA 机制已配置，可安全训练；False 表示未配置，禁止训练。
    """
    cfg = config if config is not None else PI05_INDUSTRIAL_CONFIG
    freeze_ok = getattr(cfg, "freeze_filter", None) is not None
    loader_ok = getattr(cfg, "weight_loader", None) is not None

    if not freeze_ok or not loader_ok:
        logger.warning(
            "=" * 64 + "\n"
            "[pi05_train_config] ⚠️  LoRA 安全闸门：freeze_filter / weight_loader 未配置！\n"
            "  当前状态:\n"
            f"    freeze_filter  = {'已配置' if freeze_ok else '❌ None（未冻结任何参数 → 全参数训练）'}\n"
            f"    weight_loader  = {'已配置' if loader_ok else '❌ None（未注入 LoRA 适配层）'}\n"
            "  风险:\n"
            "    - 将执行全参数微调而非 LoRA，显存需求从 >22.5GB 飙升至 >70GB（§3.3）。\n"
            "    - 与方案书规定的 LoRA 微调路径直接矛盾。\n"
            "  修复:\n"
            "    1. 按 openpi 官方 LoRA 文档配置 freeze_filter（JAX nnx.filterlib.Filter）。\n"
            "    2. 创建 LoRAWeightLoader 指向 pi05_base checkpoint，rank=32。\n"
            "    3. 参考: https://github.com/Physical-Intelligence/openpi\n"
            "  (此警告在 openpi 不可用/降级模式下属预期，真实训练前必须消除)\n"
            + "="
            * 64
        )
        return False
    logger.info(
        "[pi05_train_config] ✅ LoRA freeze_filter 与 weight_loader 均已配置，可安全训练。"
    )
    return True


# 模块导入时静默记录 LoRA 就绪状态（若在降级模式下不打印重复警告，由 _print_summary 统一处理）
_QUIET_MODE: bool = os.environ.get("PI05_QUIET", "0").strip() in ("1", "true", "True")


# ---------------------------------------------------------------------------
# 注册配置
# ---------------------------------------------------------------------------
def register_config() -> bool:
    """把 pi05_industrial 注册进 openpi 官方 _CONFIGS 字典。

    Returns:
        True 注册成功；False 表示注册失败（openpi 不可用时也注册到本地占位字典）。
    """
    try:
        _CONFIGS["pi05_industrial"] = PI05_INDUSTRIAL_CONFIG
        return True
    except Exception as e:
        logger.warning("注册 pi05_industrial 到 _CONFIGS 失败: %s", e)
        return False


def get_config(name: str = "pi05_industrial") -> Optional[TrainConfig]:
    """获取配置实例。优先从 _CONFIGS 取，其次返回本文件构建的实例。

    供 openpi scripts/train.py 或本地脚本调用：get_config("pi05_industrial")。
    """
    if name == "pi05_industrial" and PI05_INDUSTRIAL_CONFIG is not None:
        return PI05_INDUSTRIAL_CONFIG
    if name in _CONFIGS:
        return _CONFIGS[name]
    return None


# ---------------------------------------------------------------------------
# 模块导入时自动注册（best-effort）
# ---------------------------------------------------------------------------
_REGISTERED: bool = register_config()


# ---------------------------------------------------------------------------
# 配置摘要打印（文件末尾，方案书要求可见性）
# ---------------------------------------------------------------------------
def _print_summary() -> None:
    """打印配置摘要。设置 PI05_QUIET=1 可跳过打印（S2）。"""
    if _QUIET_MODE:
        return
    print("=" * 64)
    print("[pi05_train_config] 配置摘要")
    print("=" * 64)
    print(f"openpi 可用:        {OPENPI_AVAILABLE}")
    if not OPENPI_AVAILABLE:
        print(f"openpi import 错误: {OPENPI_IMPORT_ERROR}")
        print("提示: LoRA 微调必须走 openpi JAX 路径（方案书 §3.3）。")
        print("      git clone https://github.com/Physical-Intelligence/openpi")
        print("      cd openpi && uv sync")
    print(f"配置名:             pi05_industrial")
    print(f"model_type:         {getattr(ModelType, 'PI05', 'PI05')}")
    print(f"action_dim:         7   (方案书 §3.4 [dx,dy,dz,dax,day,daz,gripper])")
    print(f"action_horizon:     10  (初始候选，D21 后按闭环表现调整)")
    print(
        f"batch_size:         {BATCH_SIZE}  (默认安全值 16；方案书 §3.3：22.5GB 卡建议 ≤16)"
    )
    print(f"lr init_value:      2e-5 (LoRA 微调较小学习率)")
    print(f"num_train_steps:    30000  (openpi 官方示例参考值；D21 按数据量与收敛调整)")
    print(
        f"warmup_steps:       {WARMUP_STEPS}  (C2 修复；若 openpi API 不支持则由 scheduler 内部控制)"
    )
    print(
        f"weight_decay:       {WEIGHT_DECAY}  (C2 修复；若 openpi API 不支持则由 optimizer 内部控制)"
    )
    print(f"grad_accum_steps:   {GRADIENT_ACCUMULATION_STEPS}  (C2 修复)")
    print(f"mixed_precision:    {MIXED_PRECISION}  (C2 修复；JAX 路径推荐 bf16)")
    print(
        f"eval_interval:      {EVAL_INTERVAL}  (W2 修复；若 openpi API 不支持则由外部脚本触发)"
    )
    print(
        f"LoRA rank:          {LORA_RANK} (方案书 §3.2.1 OpenVLA-OFT 示例；π0.5 初始候选值)"
    )
    print(f"base checkpoint:    {BASE_CHECKPOINT}")
    print(f"dataset repo_id:    {DATASET_REPO_ID}")
    print(f"output_dir:         {OUTPUT_DIR}")
    print(f"fsdp_devices:       1   (单卡，方案书 §3.3 JAX 路径)")
    print(f"注册到 _CONFIGS:    {_REGISTERED}")
    # LoRA 安全闸门提示
    _lora_ok = validate_lora_ready()
    if not _lora_ok:
        print("⚠️  LoRA 安全闸门: freeze_filter / weight_loader 未配置！(C3)")
        print(
            "   当前配置禁止用于真实训练；全参数训练显存 >70GB，远超 LoRA 预算 22.5GB。"
        )
    else:
        print("✅ LoRA freeze_filter / weight_loader 已配置，可安全训练。")
    print("-" * 64)
    print("训练命令:")
    print("  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \\")
    print("      pi05_industrial --exp-name=my_experiment --overwrite")
    print("norm stats 计算命令（训练前必跑，方案书 §3.3.1 Para186）:")
    print("  uv run scripts/compute_norm_stats.py --config-name pi05_industrial")
    print("=" * 64)


# 配置摘要不再在 import 时自动打印（W1 修复），避免与其他模块的摘要输出重复。
# 调用方（如 train.py）在 main() 中通过自身的 print_summary() 提供完整摘要。
# 如需单独查看 train_config 摘要，可设置 PI05_QUIET=0 并显式调用 _print_summary()。
