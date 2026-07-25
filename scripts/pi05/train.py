"""scripts/pi05/train.py

π0.5 模型 LoRA 微调训练入口脚本（封装 openpi 官方 scripts/train.py）。

负责人：E（π0.5/openpi）
方案书出处：
- §3.3 / §3.3.1：π0.5 适配流程（JAX 路径、LoRA、norm stats、动作块适配）。
- §6.3：主 VLA 训练流程（官方复现 → 接口烟测 → 首轮微调 → 定向回灌 → 冻结）。
- §3.3.1 Para186：本项目自有 norm stats，训练前必跑 compute_norm_stats。

红线要求（Zero-Tolerance）：
1. 环境变量时序：XLA_PYTHON_CLIENT_MEM_FRACTION 必须在所有 JAX/openpi import 之前设置，
   避免因 JAX 提前初始化导致显存限制策略失效。
2. 标识符严格匹配：配置名 pi05_industrial、冻结 filter 等与 train_config.py 完全一致。
3. CPU/Mock 环境兼容性：--mock 模式下豁免或生成 Mock Stats，无 GPU/openpi 时不崩溃。
4. 路径安全规范：checkpoint 与 assets 路径通过 CLI 参数或环境变量获取，不写死绝对路径。

用法：
    # 真实训练（GPU 服务器，openpi 已安装）
    python scripts/pi05/train.py --config-name pi05_industrial \\
        --exp-name my_experiment --overwrite

    # Mock 骨架验证（CPU，无 openpi 也能跑）
    python scripts/pi05/train.py --mock

    # 从 checkpoint 恢复训练
    python scripts/pi05/train.py --config-name pi05_industrial \\
        --exp-name my_experiment --resume

环境变量：
    XLA_PYTHON_CLIENT_MEM_FRACTION: JAX 显存预占比例，默认 0.9（可被外部覆盖）
    PI05_CHECKPOINT_DIR: 检查点存储根目录（覆盖 config.checkpoint_base_dir）
    PI05_ASSETS_DIR: 资产文件存储根目录（覆盖 config.assets_base_dir）
    PI05_OPENPI_REPO: openpi 仓库根目录（用于定位官方 scripts/train.py，可编辑安装时可不设）
    PI05_QUIET: 静默模式（=1 时 train_config 不打印摘要）
"""
from __future__ import annotations

# ===========================================================================
# 红线 1：XLA_PYTHON_CLIENT_MEM_FRACTION 必须在所有 JAX/openpi import 之前设置
# ===========================================================================
# 此处仅 import os，不 import 任何 JAX/openpi 相关模块，确保环境变量先生效。
import os as _os

# 默认 0.9（方案书 §3.3 + Dockerfile.pi05 已设 0.9）；setdefault 允许外部环境变量覆盖
_DEFAULT_MEM_FRACTION = "0.9"
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", _DEFAULT_MEM_FRACTION)

# 提前探测 --quiet，避免 train_config 在 import 时打印摘要（train_config 读 PI05_QUIET）
# 注意：必须先 import sys 才能访问 argv；此处只做字符串探测，不解析 argparse。
import sys as _sys
if "--quiet" in _sys.argv:
    _os.environ.setdefault("PI05_QUIET", "1")

# Windows 控制台默认 GBK(cp936) 编码，无法编码 train_config.py 中的 emoji（⚠️/✅），
# 会导致 UnicodeEncodeError 崩溃。此处不强制改编码（否则中文会乱码），而是用 errors="replace"
# 把无法编码的字符替换为 '?'，保证不崩溃且中文正常显示。
# Linux/Docker 默认 UTF-8，reconfigure 无副作用。
# 必须在 import configs.pi05.train_config 之前完成（train_config 的 _print_summary 在 import 时执行）。
try:
    _sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    _sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    # 旧版 Python 或已被包装的 stream 不支持 reconfigure，忽略
    pass

# ===========================================================================
# 标准库 import（不触发 JAX 初始化）
# ===========================================================================
import argparse
import dataclasses
import importlib.util
import json
import logging
import pathlib
from typing import Any

# ===========================================================================
# 确保项目根目录在 sys.path 中（脚本可能从任意目录运行）
# ===========================================================================
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# ===========================================================================
# 红线 2：显式 import configs.pi05.train_config 触发配置注册副作用
# ===========================================================================
# 该 import 会：
#   1. 尝试 import openpi（可用时用官方类，不可用时降级为占位 dataclass）
#   2. 构建 PI05_INDUSTRIAL_CONFIG 实例
#   3. 把 pi05_industrial 注册进 _CONFIGS（openpi 官方注册表或本地占位表）
import configs.pi05.train_config as pi05_config  # noqa: E402


# ===========================================================================
# 日志
# ===========================================================================
logger = logging.getLogger("pi05_train")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][pi05_train] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ===========================================================================
# 常量（标识符严格匹配 train_config.py，严禁修改）
# ===========================================================================
DEFAULT_CONFIG_NAME: str = "pi05_industrial"
DEFAULT_MOCK_EXP_NAME: str = "mock_experiment"
DEFAULT_MEM_FRACTION: str = _DEFAULT_MEM_FRACTION  # S2：统一引用 L45 常量，避免重复定义
# norm stats 文件名（与 openpi/shared/normalize.py save() 一致）
NORM_STATS_FILENAME: str = "norm_stats.json"


# ===========================================================================
# 配置获取
# ===========================================================================
def get_config(config_name: str) -> Any:
    """获取完整配置对象。

    红线 2：标识符严格匹配 train_config.py。
    优先调用 openpi.training.config.get_config（方案书 §3.3 JAX 路径），
    openpi 不可用时降级到 train_config.py 本地 get_config（占位 dataclass）。
    """
    if pi05_config.OPENPI_AVAILABLE:
        try:
            from openpi.training.config import get_config as _openpi_get_config
            cfg = _openpi_get_config(config_name)
            if cfg is not None:
                logger.info("从 openpi.training.config.get_config 加载配置: %s", config_name)
                return cfg
            logger.warning("openpi get_config 返回 None，降级到 train_config.get_config")
        except Exception as e:
            logger.warning("openpi get_config 失败，降级到 train_config.get_config: %s", e)
    # 降级：train_config.py 本地 get_config
    cfg = pi05_config.get_config(config_name)
    if cfg is None:
        logger.error("配置 '%s' 未找到（openpi 可用=%s）", config_name, pi05_config.OPENPI_AVAILABLE)
    return cfg


# ===========================================================================
# 路径计算（红线 4：路径通过环境变量获取，不写死）
# ===========================================================================
def get_assets_dirs(config: Any, config_name: str) -> pathlib.Path:
    """获取 assets 目录路径。

    优先级：
    1. PI05_ASSETS_DIR 环境变量（覆盖 config.assets_base_dir）
    2. config.assets_dirs 属性（openpi 官方 TrainConfig 的 property）
    3. 降级默认 ./data/fixtures
    """
    assets_base = _os.environ.get("PI05_ASSETS_DIR")
    if assets_base:
        return pathlib.Path(assets_base).resolve() / config_name
    ad = getattr(config, "assets_dirs", None)
    if ad is not None:
        # W3 修复：使用 str() 统一处理 Path/str 两种类型，避免 Path(Path) 冗余包装
        return pathlib.Path(str(ad)).resolve()
    # 降级（占位 config 无 assets_dirs 属性）
    return (pathlib.Path(".") / "data" / "fixtures").resolve()


def get_norm_stats_path(config: Any, config_name: str) -> pathlib.Path:
    """获取 norm_stats.json 路径，严格对齐 compute_norm_stats.py 输出约定。

    compute_norm_stats.py 中：
        output_path = config.assets_dirs / data_config.repo_id
        normalize.save(output_path, norm_stats)  # 写入 output_path/norm_stats.json

    故完整路径为：<assets_dirs>/<repo_id>/norm_stats.json
    """
    assets_dirs = get_assets_dirs(config, config_name)
    repo_id = getattr(config.data, "repo_id", None) or "fake"
    return assets_dirs / repo_id / NORM_STATS_FILENAME


# ===========================================================================
# 训练前检查（Pre-flight Checks）
# ===========================================================================
def check_norm_stats(config: Any, config_name: str) -> bool:
    """检查 norm_stats.json 是否存在（非 mock 模式必须通过）。

    方案书 §3.3.1 Para186：训练前必跑 compute_norm_stats 生成自有统计。
    """
    path = get_norm_stats_path(config, config_name)
    if path.exists():
        logger.info("✅ norm_stats.json 已找到: %s", path)
        return True
    logger.error("❌ norm_stats.json 不存在: %s", path)
    logger.error("   方案书 §3.3.1 Para186 要求训练前必跑 compute_norm_stats。")
    logger.error("   请先运行: python scripts/pi05/compute_norm_stats.py --config-name %s", config_name)
    return False


def generate_mock_norm_stats(config: Any, config_name: str) -> pathlib.Path:
    """生成符合格式要求的 Mock norm_stats.json。

    红线 3：--mock 模式可选策略 2，自动生成 Mock Stats 防止骨架测试因缺少统计文件而断言失败。

    格式对齐 openpi/shared/normalize.py 的 _NormStatsDict：
        {"norm_stats": {"state": {mean,std,q01,q99}, "actions": {mean,std,q01,q99}}}
    """
    path = get_norm_stats_path(config, config_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 维度从 config.model 读取（红线 2：严格匹配 train_config.py 定义）
    action_dim = int(getattr(config.model, "action_dim", 7))  # 方案书 §3.4：7 维动作
    # S1 修复：state 维度从 train_config 常量读取，不再硬编码
    # Franka 7-DOF + 1 gripper = 8（convert_openpi.py DEFAULT_STATE_DIM；
    # train_config.STATE_DIM 为唯一真相源）
    state_dim = int(getattr(pi05_config, "STATE_DIM", 8))

    mock_stats = {
        "norm_stats": {
            "state": {
                "mean": [0.0] * state_dim,
                "std": [1.0] * state_dim,
                "q01": [-1.0] * state_dim,
                "q99": [1.0] * state_dim,
            },
            "actions": {
                "mean": [0.0] * action_dim,
                "std": [1.0] * action_dim,
                "q01": [-1.0] * action_dim,
                "q99": [1.0] * action_dim,
            },
        }
    }
    path.write_text(json.dumps(mock_stats, indent=2), encoding="utf-8")
    logger.info("[MOCK] 已生成 mock norm_stats.json: %s (state_dim=%d, action_dim=%d)",
                path, state_dim, action_dim)
    return path


# ===========================================================================
# 官方训练逻辑调用（Python 内部 import，严禁 subprocess）
# ===========================================================================
def _load_module_from_path(module_name: str, file_path: pathlib.Path):
    """通过文件路径加载 Python 模块（用于 import 官方 scripts/train.py）。"""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {file_path} 创建模块 spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_openpi_train_module():
    """定位并 import openpi 官方 scripts/train.py 模块。

    查找顺序（红线 4：路径可配置）：
    1. PI05_OPENPI_REPO 环境变量 → <repo>/scripts/train.py
    2. 通过 openpi 包位置推断：__init__.py → openpi → src → <repo>/scripts/train.py

    Returns:
        官方 train 模块（含 main(config) 函数）；未找到返回 None。
    """
    # 1. 显式环境变量
    repo_env = _os.environ.get("PI05_OPENPI_REPO")
    if repo_env:
        train_path = pathlib.Path(repo_env) / "scripts" / "train.py"
        if train_path.exists():
            logger.info("从 PI05_OPENPI_REPO 定位官方 train.py: %s", train_path)
            return _load_module_from_path("openpi_train_script", train_path)
        logger.warning("PI05_OPENPI_REPO=%s 但 scripts/train.py 不存在", repo_env)

    # 2. 通过 openpi 包位置推断
    try:
        # S3：此时 XLA_PYTHON_CLIENT_MEM_FRACTION 已在模块顶部（L46）设置完毕，JAX 初始化安全
        import openpi  # noqa: PLC0415
        pkg_path = pathlib.Path(openpi.__file__).resolve()
        # openpi 包结构：<repo>/src/openpi/__init__.py
        # parents: [0]=openpi, [1]=src, [2]=repo_root
        repo_root = pkg_path.parents[2]
        train_path = repo_root / "scripts" / "train.py"
        if train_path.exists():
            logger.info("从 openpi 包位置推断官方 train.py: %s", train_path)
            return _load_module_from_path("openpi_train_script", train_path)
        logger.warning("openpi 包已安装但 %s/scripts/train.py 不存在（可能为 wheel 安装）", repo_root)
    except ImportError:
        pass

    return None


# ===========================================================================
# 配置覆盖（关键参数透传）
# ===========================================================================
def apply_overrides(config: Any, args: argparse.Namespace) -> Any:
    """将 CLI 参数与环境变量应用到 config，返回新 config（dataclasses.replace）。

    透传参数：exp_name, overwrite, resume（官方 TrainConfig 字段）
    环境变量覆盖：checkpoint_base_dir, assets_base_dir
    """
    overrides: dict[str, Any] = {
        "exp_name": args.exp_name,
        "overwrite": args.overwrite,
        "resume": args.resume,
    }

    # 红线 4：路径通过环境变量获取
    ckpt_dir = _os.environ.get("PI05_CHECKPOINT_DIR")
    if ckpt_dir:
        overrides["checkpoint_base_dir"] = ckpt_dir
    assets_dir = _os.environ.get("PI05_ASSETS_DIR")
    if assets_dir:
        overrides["assets_base_dir"] = assets_dir

    # 仅保留 config 实际拥有的字段（占位 dataclass 可能缺 checkpoint_base_dir/assets_base_dir）
    if dataclasses.is_dataclass(config):
        valid_fields = {f.name for f in dataclasses.fields(config)}
        valid_overrides = {k: v for k, v in overrides.items() if k in valid_fields}
        skipped = {k: v for k, v in overrides.items() if k not in valid_fields}
        if skipped:
            logger.warning("以下覆盖字段在当前 config 中不存在，已跳过: %s（W2 修复：升级为 WARNING 确保可见）",
                       list(skipped.keys()))
        if not valid_overrides:
            return config
        return dataclasses.replace(config, **valid_overrides)

    # 非 dataconfig 兜底（不应发生）
    return config


def apply_mock_data(config: Any) -> Any:
    """--mock 模式：将 data 替换为 openpi FakeDataConfig。

    红线 3：当启用 --mock 模式时，必须将 dataset/data 相关配置替换为 openpi 自带的
    FakeDataConfig，以实现无真实数据环境下的流程测试。
    """
    if not pi05_config.OPENPI_AVAILABLE:
        logger.info("[MOCK] openpi 不可用，跳过 FakeDataConfig 替换（使用 train_config 占位 data）")
        return config
    try:
        from openpi.training.config import FakeDataConfig  # noqa: PLC0415
        new_config = dataclasses.replace(config, data=FakeDataConfig())
        logger.info("[MOCK] 已将 data 替换为 FakeDataConfig（openpi 自带，无需真实数据/norm_stats）")
        return new_config
    except Exception as e:
        logger.warning("[MOCK] 替换 FakeDataConfig 失败，保留原 data: %s", e)
        return config


# ===========================================================================
# 配置摘要打印
# ===========================================================================
def print_summary(config: Any, config_name: str, args: argparse.Namespace) -> None:
    """训练启动前打印配置摘要（方案书要求可见性）。

    至少包含：Config 名称、LoRA Rank 值、Batch Size、Total Steps、Memory Fraction。
    """
    mem_frac = _os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", DEFAULT_MEM_FRACTION)
    lora_rank = pi05_config.LORA_RANK
    batch_size = getattr(config, "batch_size", "?")
    num_steps = getattr(config, "num_train_steps", "?")
    warmup = getattr(config, "warmup_steps", pi05_config.WARMUP_STEPS)
    weight_decay = getattr(config, "weight_decay", pi05_config.WEIGHT_DECAY)
    grad_accum = getattr(config, "gradient_accumulation_steps", pi05_config.GRADIENT_ACCUMULATION_STEPS)
    mixed_prec = getattr(config, "mixed_precision", pi05_config.MIXED_PRECISION)
    eval_interval = getattr(config, "eval_interval", pi05_config.EVAL_INTERVAL)
    fsdp = getattr(config, "fsdp_devices", "?")
    exp_name = getattr(config, "exp_name", args.exp_name)
    action_dim = getattr(config.model, "action_dim", 7)
    action_horizon = getattr(config.model, "action_horizon", 10)

    # LoRA 冻结状态（方案书 §3.3.1 安全闸门）
    freeze_filter = getattr(config, "freeze_filter", None)
    weight_loader = getattr(config, "weight_loader", None)
    lora_ready = freeze_filter is not None and weight_loader is not None

    assets_dirs = get_assets_dirs(config, config_name)
    norm_stats_path = get_norm_stats_path(config, config_name)

    print("=" * 72)
    print("[pi05_train] 训练配置摘要")
    print("=" * 72)
    print(f"  Config 名称:        {config_name}")
    print(f"  实验名称 (exp_name): {exp_name}")
    print(f"  openpi 可用:        {pi05_config.OPENPI_AVAILABLE}")
    print(f"  Mock 模式:          {args.mock}")
    print("-" * 72)
    print(f"  LoRA Rank:          {lora_rank}  (方案书 §3.2.1 OpenVLA-OFT 示例；π0.5 初始候选值)")
    print(f"  action_dim:         {action_dim}  (方案书 §3.4 [dx,dy,dz,dax,day,daz,gripper])")
    print(f"  action_horizon:     {action_horizon}  (初始候选，D21 后按闭环表现调整)")
    print(f"  Batch Size:         {batch_size}  (方案书 §3.3：22.5GB 卡建议 ≤16)")
    print(f"  Total Steps:        {num_steps}  (openpi 官方示例参考值；D21 按数据量调整)")
    print(f"  Warmup Steps:       {warmup}")
    print(f"  Weight Decay:       {weight_decay}")
    print(f"  Grad Accum Steps:   {grad_accum}")
    print(f"  Mixed Precision:    {mixed_prec}  (JAX 路径推荐 bf16，方案书 §3.3)")
    print(f"  Eval Interval:      {eval_interval}")
    print(f"  FSDP Devices:       {fsdp}  (单卡=1，方案书 §3.3 JAX 路径)")
    print("-" * 72)
    print(f"  Memory Fraction:    {mem_frac}  (XLA_PYTHON_CLIENT_MEM_FRACTION)")
    print(f"  Assets Dirs:        {assets_dirs}")
    print(f"  Norm Stats Path:    {norm_stats_path}")
    print(f"  Base Checkpoint:    {pi05_config.BASE_CHECKPOINT}")
    print(f"  Dataset repo_id:    {pi05_config.DATASET_REPO_ID}")
    print("-" * 72)
    if lora_ready:
        print(f"  LoRA 冻结状态:      ✅ freeze_filter + weight_loader 已配置")
    else:
        print(f"  LoRA 冻结状态:      ⚠️  freeze_filter/weight_loader 未配置（C3 安全闸门）")
        print(f"                       当前为全参数训练配置，显存需求 >70GB（§3.3）")
        if not args.mock:
            print(f"                       真实训练前必须配置 LoRA 机制；--mock 模式可豁免。")
    print("-" * 72)
    print(f"  Overwrite:          {args.overwrite}")
    print(f"  Resume:             {args.resume}")
    print(f"  Quiet:              {args.quiet}")
    print("=" * 72)


# ===========================================================================
# CLI 接口设计
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    必需参数：
        --config-name: 配置名称，默认 pi05_industrial
        --exp-name: 实验名称，非 mock 模式必须显式指定

    可选参数：
        --mock: 启用 Mock 模式（CPU/无 GPU 骨架验证）
        --overwrite: 覆盖已有实验
        --resume: 从 checkpoint 恢复训练
        --quiet: 静默模式
        --mock-stats: Mock 模式 norm_stats 处理策略（skip|generate）
    """
    parser = argparse.ArgumentParser(
        description="π0.5 模型 LoRA 微调训练入口（封装 openpi 官方 scripts/train.py）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 必需参数
    parser.add_argument(
        "--config-name", default=DEFAULT_CONFIG_NAME,
        help="配置名称，必须与 train_config.py 中定义一致（默认 pi05_industrial）。",
    )
    parser.add_argument(
        "--exp-name", default=None,
        help="实验名称，非 mock 模式必须显式指定（用于命名 checkpoint 目录）。",
    )
    # 可选参数
    parser.add_argument(
        "--mock", action="store_true",
        help="启用 Mock 模式：CPU/无 GPU 环境验证训练骨架（替换 FakeDataConfig，豁免 norm_stats）。",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已有实验 checkpoint 目录。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从最近 checkpoint 恢复训练。",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式，减少输出信息。",
    )
    parser.add_argument(
        "--mock-stats", choices=["skip", "generate"], default="skip",
        help="Mock 模式 norm_stats 处理策略：skip=跳过校验；generate=生成 Mock 统计文件。",
    )
    return parser.parse_args()


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    """训练主入口。返回退出码（0=成功）。"""
    args = parse_args()

    # 静默模式
    if args.quiet:
        logger.setLevel(logging.WARNING)

    # ---- 1. 获取配置（红线 2：标识符严格匹配 train_config.py）----
    logger.info("加载配置: %s", args.config_name)
    config = get_config(args.config_name)
    if config is None:
        logger.error("配置加载失败，退出。")
        return 1

    # ---- 2. 校验 exp_name（非 mock 模式必须显式指定）----
    if args.exp_name is None:
        if args.mock:
            args.exp_name = DEFAULT_MOCK_EXP_NAME
            logger.info("[MOCK] 未指定 --exp-name，使用默认值: %s", DEFAULT_MOCK_EXP_NAME)
        else:
            logger.error("非 mock 模式下 --exp-name 必须显式指定。")
            return 1

    # ---- 3. 应用 CLI 参数与环境变量覆盖（关键参数透传）----
    config = apply_overrides(config, args)
    logger.info("已应用 CLI 覆盖: exp_name=%s, overwrite=%s, resume=%s",
                args.exp_name, args.overwrite, args.resume)

    # ---- 4. Mock 模式：替换 data 为 FakeDataConfig（红线 3）----
    if args.mock:
        config = apply_mock_data(config)

    # ---- 5. 训练前检查（Pre-flight Checks）----
    if args.mock:
        if args.mock_stats == "generate":
            logger.info("[MOCK] 策略=generate：生成 Mock norm_stats.json")
            generate_mock_norm_stats(config, args.config_name)
        else:
            logger.info("[MOCK] 策略=skip：跳过 norm_stats.json 校验")
    else:
        # 非 mock 模式：norm_stats.json 必须存在（方案书 §3.3.1 Para186）
        if not check_norm_stats(config, args.config_name):
            return 1

    # ---- 6. 打印训练配置摘要（--quiet 时跳过）----
    if not args.quiet:
        print_summary(config, args.config_name, args)

    # ---- 7. 调用官方训练逻辑（Python 内部 import，严禁 subprocess）----
    if not pi05_config.OPENPI_AVAILABLE:
        # openpi 不可用：仅 --mock 模式可继续（骨架验证）
        if args.mock:
            logger.info("=" * 60)
            logger.info("[MOCK] 骨架验证完成。")
            logger.info("[MOCK] 已验证：配置加载 ✓ / 参数透传 ✓ / Pre-flight 检查 ✓ / 摘要打印 ✓")
            logger.info("[MOCK] 真实训练需要 openpi（方案书 §3.3 JAX 路径）：")
            logger.info("[MOCK]   git clone https://github.com/Physical-Intelligence/openpi")
            logger.info("[MOCK]   cd openpi && uv sync && uv pip install -e .")
            logger.info("=" * 60)
            return 0
        logger.error("openpi 不可用，无法执行真实训练。使用 --mock 验证骨架。")
        if pi05_config.OPENPI_IMPORT_ERROR:
            logger.error("import 错误: %s", pi05_config.OPENPI_IMPORT_ERROR)
        return 1

    # openpi 可用：定位并调用官方 scripts/train.py main(config)
    train_module = load_openpi_train_module()
    if train_module is None:
        logger.error("openpi 已安装但无法定位官方 scripts/train.py。")
        logger.error("请设置 PI05_OPENPI_REPO 环境变量指向 openpi 仓库根目录。")
        if args.mock:
            logger.info("[MOCK] 骨架验证部分完成（无法定位官方 train.py，跳过训练循环）。")
            return 0
        return 1

    # 红线 3：关键参数透传给官方训练函数
    logger.info("调用 openpi 官方 scripts/train.py main(config) 开始训练...")
    try:
        train_module.main(config)
    except KeyboardInterrupt:
        logger.warning("训练被用户中断（KeyboardInterrupt）。")
        return 130
    except Exception as e:
        logger.error("训练异常退出: %s", e)
        raise

    logger.info("训练完成。")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
