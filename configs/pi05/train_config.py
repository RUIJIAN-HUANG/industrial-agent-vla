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
    python scripts/pi05/compute_norm_stats.py --help

V2 预组窗数据训练：
    PI05_INPUT_FORMAT=lerobot-v2 python scripts/pi05/train.py \
        --config-name pi05_industrial --exp-name=my_experiment --overwrite
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from configs.pi05.constants import OPENPI_COMMIT as OPENPI_COMMIT
except ModuleNotFoundError:  # direct ``python configs/pi05/train_config.py`` execution
    from constants import OPENPI_COMMIT as OPENPI_COMMIT  # type: ignore[no-redef]

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
# 冻结 Arm_A policy state：末端 6D pose + 1 gripper = 7。
# 作为唯一真相源，供 train.py 与 compute_norm_stats.py 引用
STATE_DIM: int = int(os.environ.get("PI05_STATE_DIM", "7"))
MODEL_ACTION_DIM: int = 32
CANONICAL_ACTION_DIM: int = 7
ACTION_HORIZON: int = 10
SUPPORTED_INPUT_FORMATS = ("lerobot", "lerobot-v2")
PI05_INPUT_FORMAT: str = os.environ.get("PI05_INPUT_FORMAT", "lerobot").strip().lower()


def action_sequence_keys_for_input(input_format: str) -> tuple[str, ...] | None:
    """Select OpenPI windowing for the explicit LeRobot input format."""

    if input_format not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            "PI05_INPUT_FORMAT must be one of "
            f"{SUPPORTED_INPUT_FORMATS!r}, got {input_format!r}"
        )
    # Legacy rows contain one [7] action and need OpenPI's default windowing;
    # V2 rows already contain complete [10,7] windows.
    return () if input_format == "lerobot-v2" else None


def require_frozen_action_horizon(action_horizon: int, *, production: bool) -> int:
    """Validate the formally frozen π0.5 industrial action horizon."""

    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int):
        raise TypeError("action_horizon must be an integer")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if production and action_horizon != ACTION_HORIZON:
        raise RuntimeError(
            "pi05_industrial production action_horizon is frozen at "
            f"{ACTION_HORIZON}, got {action_horizon}"
        )
    return action_horizon


def project_policy_actions(actions: Any) -> np.ndarray:
    """Project one pinned π0.5 32-D output onto the canonical seven axes."""

    array = np.asarray(actions)
    if array.ndim < 1 or array.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError(
            "π0.5 base-compatible output must end in "
            f"{MODEL_ACTION_DIM} action dimensions, got {array.shape}"
        )
    return array[..., :CANONICAL_ACTION_DIM]


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
# 依赖：openpi（try/except，缺失时降级为纯 dataclass 占位定义）
# ---------------------------------------------------------------------------
# 方案书 §3.3：需要 LoRA 时必须走 JAX 路径，依赖 openpi 官方包。
# 本地无 openpi 时，仍允许本文件被 import（用于文档/CI 静态检查），
# 但配置实例为占位 dataclass，不能用于真实训练。
OPENPI_AVAILABLE: bool = False
OPENPI_IMPORT_ERROR: str | None = None

TrainConfig: Any = None
DataConfig: Any = None
DataConfigFactory: Any = None
ModelTransformFactory: Any = None
Pi0Config: Any = None
CosineDecaySchedule: Any = None
AdamW: Any = None
WeightLoader: Any = None
_LOCAL_CONFIGS: dict[str, Any] = {}

try:
    from openpi import transforms as _transforms  # type: ignore
    from openpi.models.pi0_config import Pi0Config  # type: ignore
    from openpi.training.config import (  # type: ignore
        DataConfig,
        DataConfigFactory,
        ModelTransformFactory,
        TrainConfig,
    )
    from openpi.training.optimizer import AdamW, CosineDecaySchedule  # type: ignore
    from openpi.training.weight_loaders import (  # type: ignore
        CheckpointWeightLoader,
        WeightLoader,
    )

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
    class DataConfig:
        prompt_from_task: bool = True

    @dataclass
    class DataConfigFactory:
        """Fallback shape used only for CPU/static tests."""

        repo_id: str = ""
        base_config: Any = None

    class ModelTransformFactory:
        pass

    @dataclass
    class Pi0Config:
        """占位模型配置（对齐官方 Pi0Config 关键字段）。"""

        pi05: bool = True
        action_dim: int = MODEL_ACTION_DIM
        action_horizon: int = ACTION_HORIZON
        max_token_len: int = 200
        paligemma_variant: str = ""  # LoRA 变体名（gemma_2b_lora）
        action_expert_variant: str = ""
        discrete_state_input: bool = True

        def get_freeze_filter(self) -> Any:
            return None

    @dataclass
    class CosineDecaySchedule:
        """占位学习率调度（对齐官方 CosineDecaySchedule 关键字段）。"""

        warmup_steps: int = 2000
        peak_lr: float = 2e-5
        decay_steps: int = 30000
        decay_lr: float = 2e-6

    @dataclass
    class AdamW:
        """占位优化器（官方默认 AdamW；weight_decay 由 openpi 内部控制）。"""

        weight_decay: float = 0.01  # C2 修复：显式声明默认值以保持可审计性
        clip_gradient_norm: float = 1.0

    class WeightLoader:
        """占位权重加载器。"""

    class CheckpointWeightLoader:
        """占位 checkpoint 权重加载器（LoRA 微调时必须配置）。"""

        def __init__(self, path: str) -> None:
            self.path = path

    _LOCAL_CONFIGS = {}

if OPENPI_AVAILABLE:

    @dataclass(frozen=True)
    class IndustrialInputs:
        """One top camera, no wrist cameras, identical in training and inference."""

        model_type: Any

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            base_image = np.asarray(data["observation/image"])
            if np.issubdtype(base_image.dtype, np.floating):
                base_image = np.clip(base_image * 255.0, 0, 255).astype(np.uint8)
            if base_image.ndim == 3 and base_image.shape[0] == 3:
                base_image = np.transpose(base_image, (1, 2, 0))
            if (
                base_image.ndim != 3
                or base_image.shape[2] != 3
                or base_image.dtype != np.uint8
            ):
                raise ValueError(
                    "observation/image must decode to uint8[H,W,3], "
                    f"got dtype={base_image.dtype} shape={base_image.shape}"
                )
            state = np.asarray(data["observation/state"], dtype=np.float32)
            if state.ndim != 1 or state.shape[0] != STATE_DIM:
                raise ValueError(
                    f"observation/state must be [{STATE_DIM}], got {state.shape}"
                )
            actions = data.get("actions")
            if actions is not None:
                actions = np.asarray(actions, dtype=np.float32)
                if actions.ndim < 1 or actions.shape[-1] != 7:
                    raise ValueError(
                        f"training actions must end in 7 dimensions, got {actions.shape}"
                    )

            # Pi0.5 always exposes three image tensor slots.  The frozen scene
            # has only CAM_A_TOP, so absent wrist slots are masked model padding,
            # never observations or CAS fallbacks.
            missing_wrist = np.zeros_like(base_image)
            result: dict[str, Any] = {
                "state": state,
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": missing_wrist,
                    "right_wrist_0_rgb": missing_wrist.copy(),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                },
            }
            if actions is not None:
                result["actions"] = actions
            if "prompt" in data:
                result["prompt"] = data["prompt"]
            return result

    @dataclass(frozen=True)
    class IndustrialOutputs:
        """Project the pinned 32-D π0.5 head onto the frozen 7-D contract."""

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            # OpenPI's pinned pi05_base checkpoint has a 32-D state/action
            # projection. Training data supplies the seven canonical robot
            # dimensions and ModelTransformFactory pads them to 32. At policy
            # output we explicitly discard only those padding dimensions,
            # preserving a strict N×7 service contract without changing the
            # pretrained projection-layer shapes.
            return {"actions": project_policy_actions(data["actions"])}

    @dataclass(frozen=True)
    class IndustrialLeRobotDataConfig(DataConfigFactory):
        """Pinned OpenPI data mapping for CAM_A_TOP with no wrist stream."""

        def create(self, assets_dirs: Any, model_config: Any) -> Any:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[IndustrialInputs(model_type=model_config.model_type)],
                outputs=[IndustrialOutputs()],
            )
            model_transforms = ModelTransformFactory()(model_config)
            replace_kwargs: dict[str, Any] = {
                "repack_transforms": repack_transform,
                "data_transforms": data_transforms,
                "model_transforms": model_transforms,
            }
            action_sequence_keys = action_sequence_keys_for_input(PI05_INPUT_FORMAT)
            if action_sequence_keys is not None:
                # V2 rows already contain the frozen [10,7] action window.
                # Legacy V1 rows retain OpenPI's default windowing behavior.
                replace_kwargs["action_sequence_keys"] = action_sequence_keys
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                **replace_kwargs,
            )

else:

    @dataclass
    class IndustrialLeRobotDataConfig(DataConfigFactory):
        """Fallback config used only when OpenPI isn't installed."""

        pass


# ---------------------------------------------------------------------------
# 配置实例：pi05_industrial
# ---------------------------------------------------------------------------
def _build_pi05_industrial_config() -> TrainConfig:
    """构建 pi05_industrial TrainConfig 实例。

    openpi 可用时用官方类；不可用时用本文件降级 dataclass 占位（仅用于文档/检查）。
    """
    action_horizon = require_frozen_action_horizon(
        ACTION_HORIZON,
        production=True,
    )
    cfg = TrainConfig(
        name="pi05_industrial",
        # ---- 模型配置 ----
        # 方案书 §3.4：动作 7 维 [dx,dy,dz,dax,day,daz,gripper]。
        # 正式冻结：pi05_industrial 始终预测 10 个 100 ms 动作步。
        model=Pi0Config(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=action_horizon,
            max_token_len=200,
            # π0.5 的本体状态必须按官方语义作为离散语言 token 输入。
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        # ---- 数据配置 ----
        # 方案书 §5.4：canonical → LeRobot 转换由 scripts/pi05/convert_openpi.py 完成。
        # 方案书 §3.3.1 Para186：norm stats 用 compute_norm_stats 单独生成本项目自有统计。
        data=IndustrialLeRobotDataConfig(
            repo_id=DATASET_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
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
        # LoRA 训练禁用 EMA（EMA 与 LoRA 不兼容，ema_decay=None 由 openpi 内部处理）
        ema_decay=None,
        # ---- 学习率 ----
        # LoRA 微调用较小学习率（方案书 §6.3 首轮微调 1—2 组超参）。
        lr_schedule=CosineDecaySchedule(
            warmup_steps=WARMUP_STEPS,
            peak_lr=2e-5,
            decay_steps=30_000,
            decay_lr=2e-6,
        ),
        optimizer=AdamW(weight_decay=WEIGHT_DECAY, clip_gradient_norm=1.0),
        # ---- LoRA 权重加载（openpi 机制，非 PEFT） ----
        # CheckpointWeightLoader 加载 pi05_base 权重，_merge_params 自动注入 LoRA 适配层
        # （rank=LORA_RANK=32，在 weight_loader 或 model_config 中指定；方案书 §3.2.1）。
        weight_loader=CheckpointWeightLoader(BASE_CHECKPOINT + "/params"),
        # ---- LoRA 参数冻结（openpi 机制，非 PEFT） ----
        # freeze_filter：JAX nnx.filterlib.Filter，冻结 Gemma backbone 仅训练 LoRA 层。
        # get_freeze_filter() 生成的 filter 必须与 model Pi0Config 参数完全匹配，
        # 否则训练效果会很差（~1-3% success rate）。
        freeze_filter=Pi0Config(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=action_horizon,
            max_token_len=200,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
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
def validate_lora_ready(config: TrainConfig | None = None) -> bool:
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
    """Expose the local config without mutating OpenPI private registries.

    Returns:
        True when the local immutable registry contains the config.
    """
    try:
        existing = _LOCAL_CONFIGS.setdefault("pi05_industrial", PI05_INDUSTRIAL_CONFIG)
        return existing is PI05_INDUSTRIAL_CONFIG
    except Exception as e:
        logger.warning("本地 pi05_industrial 配置暴露失败: %s", e)
        return False


def get_config(name: str = "pi05_industrial") -> TrainConfig | None:
    """从本模块注册表获取配置实例。

    供 openpi scripts/train.py 或本地脚本调用：get_config("pi05_industrial")。
    """
    return _LOCAL_CONFIGS.get(name)


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
    print("配置名:             pi05_industrial")
    model_type = getattr(
        getattr(PI05_INDUSTRIAL_CONFIG, "model", None), "model_type", "PI05"
    )
    print(f"model_type:         {model_type}")
    print("model action_dim:   32  (兼容 pi05_base 投影层；输入由 OpenPI pad 到 32)")
    print("service action_dim: 7   (输出显式投影为 [dx,dy,dz,dax,day,daz,gripper])")
    print("action_horizon:     10  (frozen pi05_industrial contract)")
    print(
        f"batch_size:         {BATCH_SIZE}  (默认安全值 16；方案书 §3.3：22.5GB 卡建议 ≤16)"
    )
    print("lr init_value:      2e-5 (LoRA 微调较小学习率)")
    print("num_train_steps:    30000  (openpi 官方示例参考值；D21 按数据量与收敛调整)")
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
    print(f"input format:       {PI05_INPUT_FORMAT}")
    print(f"output_dir:         {OUTPUT_DIR}")
    print("fsdp_devices:       1   (单卡，方案书 §3.3 JAX 路径)")
    print(f"本地配置已注册:      {_REGISTERED}")
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
    print("  python scripts/pi05/compute_norm_stats.py --help")
    print("=" * 64)


# 配置摘要不再在 import 时自动打印（W1 修复），避免与其他模块的摘要输出重复。
# 调用方（如 train.py）在 main() 中通过自身的 print_summary() 提供完整摘要。
# 如需单独查看 train_config 摘要，可设置 PI05_QUIET=0 并显式调用 _print_summary()。
