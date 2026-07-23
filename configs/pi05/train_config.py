"""configs/pi05/train_config.py

π0.5 模型 LoRA 微调训练配置（openpi / JAX 路径）。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程（JAX 路径、LeRobot、norm stats、LoRA、动作块适配）。
- Table 20 Row7（§3.3）：LoRA rank 推荐值 32。
- Table 21（§3.3）：LoRA 微调显存 >22.5GB；需要 LoRA 时必须走 JAX 路径，
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
- LoRA rank：32（方案书 Table 20 Row7）
- 显存要求：>22.5GB（方案书 Table 21）；22.5GB 卡建议 batch_size 降到 16 或 8。

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
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][pi05_config] %(message)s"))
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
# LoRA rank（方案书 Table 20 Row7；在 weight_loader / model_config 中使用）
LORA_RANK: int = int(os.environ.get("PI05_LORA_RANK", "32"))


# ---------------------------------------------------------------------------
# 依赖：openpi（try/except，缺失时降级为纯 dataclass 占位定义 + 打印提示）
# ---------------------------------------------------------------------------
# 方案书 Table 21 Row3：需要 LoRA 时必须走 JAX 路径，依赖 openpi 官方包。
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
    # 复用官方 _CONFIGS 注册表（若官方以字典形式导出）
    try:
        from openpi.training.config import _CONFIGS  # type: ignore
    except Exception:  # 官方未导出 _CONFIGS，使用本地空字典
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
        batch_size: int = 32
        num_workers: int = 2
        num_train_steps: int = 30000
        log_interval: int = 100
        save_interval: int = 1000
        keep_period: int = 5000
        overwrite: bool = False
        resume: bool = False
        wandb_enabled: bool = True
        fsdp_devices: int = 1

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

    @dataclass
    class AdamW:
        """占位优化器（官方默认 AdamW）。"""
        pass

    class WeightLoader:
        """占位权重加载器。"""
        pass

    _CONFIGS = {}

    print(
        "[pi05_train_config] WARNING: openpi 未安装或 import 失败，已降级为占位 dataclass。\n"
        f"  原因: {OPENPI_IMPORT_ERROR}\n"
        "  提示: LoRA 微调必须走 openpi JAX 路径（方案书 Table 21 Row3）。\n"
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
            action_dim=7,           # 7 维动作（方案书 §3.4）
            action_horizon=10,      # 动作块长度（初始候选，D21 后按闭环表现调整）
            max_token_len=48,       # 文本 token 最大长度
        ),
        # ---- 数据配置 ----
        # 方案书 §5.4：canonical → LeRobot 转换由 scripts/pi05/convert_openpi.py 完成。
        # 方案书 §3.3.1 Para186：norm stats 用 compute_norm_stats 单独生成本项目自有统计。
        data=LeRobotLiberoDataConfig(
            repo_id=DATASET_REPO_ID,
        ),
        # ---- 训练参数 ----
        # 方案书 Table 21：LoRA 微调显存 >22.5GB；22.5GB 卡建议降到 16 或 8。
        batch_size=32,
        num_workers=2,
        num_train_steps=30000,
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        # ---- 学习率 ----
        # LoRA 微调用较小学习率（方案书 §6.3 首轮微调 1—2 组超参）。
        lr_schedule=CosineDecaySchedule(
            init_value=2e-5,
        ),
        # ---- LoRA 相关（openpi 机制，非 PEFT） ----
        # freeze_filter：JAX nnx.filterlib.Filter，冻结 base 参数只让 LoRA 层可训练。
        #   TODO(D21): 服务器上按官方 LoRA 文档配置 freeze_filter。
        # weight_loader：LoRAWeightLoader，加载 base 权重并注入 LoRA 适配层。
        #   TODO(D21): 指向 pi05_base checkpoint（BASE_CHECKPOINT），LoRA rank=LORA_RANK。
        # 方案书 Table 20 Row7：LoRA rank=32。
        # freeze_filter=...,         # 留空，服务器上配置
        # weight_loader=...,         # 留空，指向 pi05_base checkpoint
        # ---- 其他 ----
        overwrite=True,
        wandb_enabled=True,
        fsdp_devices=1,             # 单卡用 1（方案书 Table 21：JAX 路径）
    )
    return cfg


PI05_INDUSTRIAL_CONFIG: TrainConfig = _build_pi05_industrial_config()


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
    print("=" * 64)
    print("[pi05_train_config] 配置摘要")
    print("=" * 64)
    print(f"openpi 可用:        {OPENPI_AVAILABLE}")
    if not OPENPI_AVAILABLE:
        print(f"openpi import 错误: {OPENPI_IMPORT_ERROR}")
        print("提示: LoRA 微调必须走 openpi JAX 路径（方案书 Table 21 Row3）。")
        print("      git clone https://github.com/Physical-Intelligence/openpi")
        print("      cd openpi && uv sync")
    print(f"配置名:             pi05_industrial")
    print(f"model_type:         {getattr(ModelType, 'PI05', 'PI05')}")
    print(f"action_dim:         7   (方案书 §3.4 [dx,dy,dz,dax,day,daz,gripper])")
    print(f"action_horizon:     10  (初始候选，D21 后按闭环表现调整)")
    print(f"batch_size:         32  (22.5GB 显存可能需降到 16 或 8)")
    print(f"lr init_value:      2e-5 (LoRA 微调较小学习率)")
    print(f"num_train_steps:    30000")
    print(f"LoRA rank:          {LORA_RANK} (方案书 Table 20 Row7)")
    print(f"base checkpoint:    {BASE_CHECKPOINT}")
    print(f"dataset repo_id:    {DATASET_REPO_ID}")
    print(f"output_dir:         {OUTPUT_DIR}")
    print(f"fsdp_devices:       1   (单卡，方案书 Table 21 JAX 路径)")
    print(f"注册到 _CONFIGS:    {_REGISTERED}")
    print("-" * 64)
    print("训练命令:")
    print("  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \\")
    print("      pi05_industrial --exp-name=my_experiment --overwrite")
    print("norm stats 计算命令（训练前必跑，方案书 §3.3.1 Para186）:")
    print("  uv run scripts/compute_norm_stats.py --config-name pi05_industrial")
    print("=" * 64)


# 文件末尾打印配置摘要
_print_summary()
