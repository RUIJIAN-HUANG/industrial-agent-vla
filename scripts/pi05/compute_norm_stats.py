"""scripts/pi05/compute_norm_stats.py

为 π0.5 训练数据集计算归一化统计量，输出 openpi 训练可直接读取的标准格式（JSON），
并打印 SHA256 校验和（供后续写入 model_manifest.yaml）。

负责人：E（π0.5/openpi）

方案书出处：
- §3.3.1 Para186：本项目自有 norm stats，不沿用 OpenVLA；训练前必跑 compute_norm_stats。
- §3.3.1：生成 LeRobot 数据集并运行 norm stats；检查每维分布、1%/99% 分位和夹爪双峰，
  发现异常先回到数据 QA。
- §3.4：动作 7 维 [dx,dy,dz,dax,day,daz,gripper]。
- §5.4：canonical → LeRobot 转换由 scripts/pi05/convert_openpi.py 完成；norm stats 单独算。
- §7.2：norm_stats_sha 用于日志定位唯一统计资产。
- §8.5 model_manifest.yaml：norm stats SHA 用于资产追溯。

字段名 / 维度唯一真相源：configs/pi05/train_config.py
  - Pi0Config.action_dim = 7（方案书 §3.4）
  - 统计键严格沿用官方 openpi scripts/compute_norm_stats.py：["state", "actions"]
  - state 维度由数据决定（convert_openpi.py DEFAULT_STATE_DIM=8，Franka 7-DOF+gripper）

输出 JSON Schema（与 src/openpi/shared/normalize.py 的 NormStats 100% 一致）：
{
  "norm_stats": {
    "state":   {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
  }
}
注：min/max 仅在终端打印供 QA（§3.3.1 每维分布检查），不写入 JSON，以保 Schema 100% 一致。

CPU 兼容：--mock 用 numpy 随机数据独立运行，无 GPU / openpi / lerobot 也能跑通。
路径安全：所有路径走 CLI 参数或环境变量，禁止写死本地绝对路径。

用法：
    # Mock 模式（本地 CPU 验证，无需真实数据 / 无 GPU）
    python scripts/pi05/compute_norm_stats.py --mock

    # 真实数据模式（LeRobot 数据集目录 或 canonical episode 目录 或 .npz）
    python scripts/pi05/compute_norm_stats.py --dataset-path /path/to/dataset

    # 指定输出 + 静默（只输出结果与 SHA256）
    python scripts/pi05/compute_norm_stats.py --dataset-path /path/to/ds \\
        --output-path ./assets/norm_stats.json --quiet
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("compute_norm_stats")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][compute_norm_stats] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 依赖：openpi.shared.normalize（可选，存在则实例化官方 NormStats 序列化）
# ---------------------------------------------------------------------------
_normalize: Any = None
OPENPI_NORMALIZE_AVAILABLE: bool = False
try:
    from openpi.shared import normalize as _normalize  # type: ignore
    OPENPI_NORMALIZE_AVAILABLE = True
except Exception:  # 本地无 openpi，降级为本地等价实现（Schema 完全相同）
    _normalize = None


# ---------------------------------------------------------------------------
# 依赖：pandas / h5py（真实数据读取，可选）
# ---------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE: bool = True
except Exception:
    PANDAS_AVAILABLE = False

try:
    import h5py  # type: ignore
    H5PY_AVAILABLE: bool = True
except Exception:
    H5PY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 常量（维度 / 字段名严格以 train_config.py 为准）
# ---------------------------------------------------------------------------
# action_dim 源：configs/pi05/train_config.py -> Pi0Config(action_dim=7)
# 默认 7；真正值在 main() 中由 _load_action_dim_from_train_config() 覆写
# （延迟到 main() 是为了在 --quiet 模式下先调整日志级别，避免泄漏进度信息）。
ACTION_DIM: int = 7

NORM_STATS_KEYS: Tuple[str, ...] = ("state", "actions")  # 官方 compute_norm_stats.py 键
DEFAULT_STATE_DIM: int = 8  # 源：convert_openpi.py DEFAULT_STATE_DIM（Franka 7-DOF+gripper）
EPS: float = 1e-6           # 数值安全：std 下限（方案书要求防除零）
MOCK_SEED: int = 42         # mock 固定种子，保证 SHA256 可复现


def _load_action_dim_from_train_config() -> int:
    """从 configs/pi05/train_config.py 读取 Pi0Config.action_dim（红线：维度严格按 train_config）。

    动态加载文件，避免 openpi 不可用时 import 失败；加载期间抑制其自带的摘要打印。
    """
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "pi05" / "train_config.py"
    if not cfg_path.exists():
        logger.warning("未找到 %s，action_dim 回退到 7", cfg_path)
        return 7
    buf = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location("_pi05_train_config_readonly", cfg_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        # 必须先注册到 sys.modules，否则 train_config.py 内的 @dataclass 装饰器
        # 在解析字段类型时会调用 sys.modules.get(cls.__module__).__dict__ 而失败。
        sys.modules[spec.name] = mod
        # 抑制 train_config 导入时的 _print_summary / openpi 不可用提示
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
        cfg = getattr(mod, "PI05_INDUSTRIAL_CONFIG", None)
        if cfg is not None and getattr(cfg, "model", None) is not None:
            dim = int(getattr(cfg.model, "action_dim", 7))
            logger.info("从 train_config.py 读取 action_dim=%d", dim)
            return dim
    except Exception as e:
        logger.warning("读取 train_config.py 失败，action_dim 回退到 7: %s", e)
    finally:
        # 清理临时模块注册，避免污染调用方 sys.modules
        sys.modules.pop("_pi05_train_config_readonly", None)
    return 7


# ---------------------------------------------------------------------------
# 本地 NormStats（openpi 不可用时的等价 dataclass，字段与官方 100% 对齐）
# ---------------------------------------------------------------------------
@dataclass
class NormStats:
    """与 openpi.shared.normalize.NormStats 字段 100% 对齐。

    字段：mean / std / q01 / q99（均为 1-D NDArray，长度 = 该键维度）。
    """
    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None  # 1% 分位
    q99: Optional[np.ndarray] = None  # 99% 分位


# ---------------------------------------------------------------------------
# 统计量计算
# ---------------------------------------------------------------------------
def compute_stats(
    arr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    key: str = "",
) -> Dict[str, np.ndarray]:
    """沿 batch/time 轴计算 mean/std/min/max/q01/q99。

    Args:
        arr: [N, D] float，N 为样本数（batch/time 展平），D 为该键维度。
        mask: [N] bool，True=有效；若提供则先过滤 padding/无效填充（防污染统计量）。
        key: 键名，仅用于错误信息。

    Returns:
        dict: mean/std/q01/q99/min/max，均为 [D] float64。
        其中 std 已做数值安全：np.maximum(std, EPS) 防除零。
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"[{key}] 期望 2-D [N, D]，实际 ndim={arr.ndim} shape={arr.shape}")

    # ---- 掩码过滤 padding（方案书：必须用掩码过滤无效填充，防污染统计量）----
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != arr.shape[0]:
            logger.warning(
                "[%s] mask 长度 %d != 样本数 %d，忽略 mask", key, mask.shape[0], arr.shape[0]
            )
        else:
            valid_before = arr.shape[0]
            arr = arr[mask]
            valid_after = arr.shape[0]
            logger.info("[%s] mask 过滤：%d -> %d 有效（剔除 padding %d）",
                        key, valid_before, valid_after, valid_before - valid_after)

    n = arr.shape[0]
    if n == 0:
        raise ValueError(f"[{key}] 有效样本数为 0，无法计算统计量")
    if n < 2:
        raise ValueError(f"[{key}] 有效样本数 {n} < 2，无法计算 std / 分位数")

    mean = arr.mean(axis=0)
    # 总体方差（ddof=0），与官方 RunningStats 的 E[x^2]-E[x]^2 一致
    std = arr.std(axis=0, ddof=0)
    # 数值安全：std 过小处加 eps 防除零（方案书红线）
    std = np.maximum(std, EPS)
    q01 = np.quantile(arr, 0.01, axis=0)
    q99 = np.quantile(arr, 0.99, axis=0)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return {"mean": mean, "std": std, "q01": q01, "q99": q99, "min": mn, "max": mx}


def build_norm_stats(stats_by_key: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, NormStats]:
    """把原始统计 dict 转为 NormStats（仅 mean/std/q01/q99，不含 min/max）。"""
    return {
        k: NormStats(
            mean=v["mean"].astype(np.float64),
            std=v["std"].astype(np.float64),
            q01=v["q01"].astype(np.float64),
            q99=v["q99"].astype(np.float64),
        )
        for k, v in stats_by_key.items()
    }


# ---------------------------------------------------------------------------
# 序列化（Schema 与 openpi.shared.normalize.serialize_json 100% 一致）
# ---------------------------------------------------------------------------
def serialize_norm_stats(norm_stats: Dict[str, NormStats]) -> str:
    """序列化为 JSON 字符串。

    openpi 可用时优先用官方 normalize.serialize_json；否则本地构造等价结构。
    输出顶层结构：{"norm_stats": {key: {"mean":[], "std":[], "q01":[], "q99":[]}}}。
    """
    if OPENPI_NORMALIZE_AVAILABLE and _normalize is not None:
        # 用官方 NormStats dataclass 实例化并序列化，确保 Schema 100% 相同
        official_ns = {
            k: _normalize.NormStats(  # type: ignore[attr-defined]
                mean=np.asarray(v.mean, dtype=np.float64),
                std=np.asarray(v.std, dtype=np.float64),
                q01=np.asarray(v.q01, dtype=np.float64),
                q99=np.asarray(v.q99, dtype=np.float64),
            )
            for k, v in norm_stats.items()
        }
        return _normalize.serialize_json(official_ns)  # type: ignore[attr-defined]

    # 本地等价序列化（结构与官方 model_dump_json(indent=2) 对齐）
    payload = {
        "norm_stats": {
            k: {
                "mean": np.asarray(v.mean, dtype=np.float64).tolist(),
                "std": np.asarray(v.std, dtype=np.float64).tolist(),
                "q01": np.asarray(v.q01, dtype=np.float64).tolist(),
                "q99": np.asarray(v.q99, dtype=np.float64).tolist(),
            }
            for k, v in norm_stats.items()
        }
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def save_norm_stats(output_path: Path, norm_stats: Dict[str, NormStats]) -> None:
    """保存 JSON 到 output_path（创建父目录）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_norm_stats(norm_stats), encoding="utf-8")


# ---------------------------------------------------------------------------
# SHA256 校验和（供 model_manifest.yaml，方案书 §8.5 / §7.2）
# ---------------------------------------------------------------------------
def compute_sha256(path: Path) -> str:
    """计算文件 bytes 的 SHA256（64 位十六进制）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Mock 数据生成（CPU 兼容验证，固定种子保证 SHA256 可复现）
# ---------------------------------------------------------------------------
def generate_mock_data() -> Dict[str, np.ndarray]:
    """生成合成随机数据，模拟 Franka 7-DOF+gripper 状态与 7 维动作。

    构造要点（方案书 §3.3.1：检查每维分布与夹爪双峰）：
      - state[:, :7]  关节角 ~ Uniform(-pi, pi)
      - state[:, 7]   夹爪 ~ Bernoulli(0.5)（双峰：0/1）
      - actions[:, :6] 末端增量 ~ Normal(0, 0.05)
      - actions[:, 6]   夹爪指令 ~ Bernoulli(0.3)（双峰）
      - 附带 mask：1000 有效帧 + 100 padding 零帧，验证掩码过滤逻辑

    Returns:
        {"state": [N_total, 8], "actions": [N_total, 7], "mask": [N_total]}
    """
    rng = np.random.default_rng(MOCK_SEED)
    n_valid = 1000
    n_pad = 100

    state_valid = np.zeros((n_valid, DEFAULT_STATE_DIM), dtype=np.float64)
    state_valid[:, :7] = rng.uniform(-np.pi, np.pi, size=(n_valid, 7))
    state_valid[:, 7] = rng.integers(0, 2, size=n_valid).astype(np.float64)

    actions_valid = np.zeros((n_valid, ACTION_DIM), dtype=np.float64)
    actions_valid[:, :6] = rng.normal(0.0, 0.05, size=(n_valid, 6))
    actions_valid[:, 6] = rng.integers(0, 2, size=n_valid).astype(np.float64)

    # padding 帧（全零，用 mask 标记为无效）
    state_pad = np.zeros((n_pad, DEFAULT_STATE_DIM), dtype=np.float64)
    actions_pad = np.zeros((n_pad, ACTION_DIM), dtype=np.float64)

    state = np.concatenate([state_valid, state_pad], axis=0)
    actions = np.concatenate([actions_valid, actions_pad], axis=0)
    mask = np.concatenate([np.ones(n_valid, dtype=bool), np.zeros(n_pad, dtype=bool)])

    logger.info(
        "mock 数据生成：state=%s actions=%s（有效 %d + padding %d，mask 验证用）",
        state.shape, actions.shape, n_valid, n_pad,
    )
    return {"state": state, "actions": actions, "mask": mask}


# ---------------------------------------------------------------------------
# 真实数据加载（LeRobot parquet / canonical episode / npz）
# ---------------------------------------------------------------------------
_STATE_CANDIDATES: Tuple[str, ...] = (
    "state", "observation.state", "robot_state", "observation_state",
)
_ACTIONS_CANDIDATES: Tuple[str, ...] = (
    "actions", "action", "action_vec",
)
_MASK_CANDIDATES: Tuple[str, ...] = (
    "mask", "pad_mask", "padding_mask", "action_mask", "state_mask",
)


def _pick_column(df_columns: Any, candidates: Tuple[str, ...]) -> Optional[str]:
    """从 DataFrame 列名中按候选顺序找到第一个匹配列。"""
    cols = set(df_columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _stack_object_array(series: Any) -> np.ndarray:
    """把 pandas Series（每元素为 list/1-D array）堆叠为 2-D float ndarray [N, D]。"""
    arrs = [np.asarray(x, dtype=np.float64) for x in series]
    if len(arrs) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(arrs, axis=0)


def _load_from_lerobot_dir(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """从 LeRobot 数据集目录读取（data/*.parquet）。

    LeRobot v2 列名可能为 observation.state / action 等，按候选名匹配。
    """
    if not PANDAS_AVAILABLE:
        return None
    data_dir = path / "data" if (path / "data").is_dir() else path
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    found_state = found_actions = False
    mask_col: Optional[str] = None

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            logger.warning("读取 parquet 失败 %s: %s", pf, e)
            continue

        s_col = _pick_column(df.columns, _STATE_CANDIDATES)
        a_col = _pick_column(df.columns, _ACTIONS_CANDIDATES)
        if s_col is None or a_col is None:
            logger.warning("%s 缺少 state/actions 列（state=%s actions=%s），跳过",
                           pf.name, s_col, a_col)
            continue
        found_state = found_actions = True

        states.append(_stack_object_array(df[s_col]))
        actions.append(_stack_object_array(df[a_col]))

        if mask_col is None:
            mask_col = _pick_column(df.columns, _MASK_CANDIDATES)
        if mask_col is not None and mask_col in df.columns:
            masks.append(np.asarray(df[mask_col].tolist(), dtype=bool))

    if not found_state or not found_actions:
        return None

    result: Dict[str, np.ndarray] = {
        "state": np.concatenate(states, axis=0) if states else np.zeros((0, 0)),
        "actions": np.concatenate(actions, axis=0) if actions else np.zeros((0, 0)),
    }
    if masks and sum(m.shape[0] for m in masks) == result["state"].shape[0]:
        result["mask"] = np.concatenate(masks, axis=0)
    return result


def _load_from_canonical_dir(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """从 canonical episode 目录读取（每个子目录含 steps.parquet / steps.hdf5）。

    与 convert_openpi.py 的 load_steps 字段名一致：robot_state / action。
    """
    episode_dirs = [p for p in sorted(path.iterdir())
                    if p.is_dir() and (p / "meta.json").exists()]
    if not episode_dirs:
        return None

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []

    for ep_dir in episode_dirs:
        steps = _load_canonical_steps(ep_dir)
        if steps is None:
            continue
        states.append(np.asarray(steps["robot_state"], dtype=np.float64))
        actions.append(np.asarray(steps["action"], dtype=np.float64))

    if not states:
        return None
    return {
        "state": np.concatenate(states, axis=0),
        "actions": np.concatenate(actions, axis=0),
    }


def _load_canonical_steps(episode_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """读取单个 canonical episode 的 steps（复用 convert_openpi.py 字段名）。"""
    parquet_path = episode_dir / "steps.parquet"
    hdf5_path = episode_dir / "steps.hdf5"

    if parquet_path.exists() and PANDAS_AVAILABLE:
        try:
            df = pd.read_parquet(parquet_path)
            robot_state = _stack_object_array(df["robot_state"]) if "robot_state" in df else None
            action = _stack_object_array(df["action"]) if "action" in df else None
            if robot_state is None or action is None:
                return None
            return {"robot_state": robot_state, "action": action}
        except Exception as e:
            logger.warning("canonical parquet 解析失败 %s: %s", parquet_path, e)
            return None

    if hdf5_path.exists() and H5PY_AVAILABLE:
        try:
            with h5py.File(hdf5_path, "r") as f:
                if "robot_state" not in f or "action" not in f:
                    return None
                return {
                    "robot_state": np.asarray(f["robot_state"], dtype=np.float64),
                    "action": np.asarray(f["action"], dtype=np.float64),
                }
        except Exception as e:
            logger.warning("canonical hdf5 解析失败 %s: %s", hdf5_path, e)
            return None

    return None


def _load_from_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """从 .npz 文件读取（state / actions / 可选 mask）。"""
    if not path.is_file() or path.suffix.lower() != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            s_col = next((c for c in _STATE_CANDIDATES if c in keys), None)
            a_col = next((c for c in _ACTIONS_CANDIDATES if c in keys), None)
            if s_col is None or a_col is None:
                return None
            result: Dict[str, np.ndarray] = {
                "state": np.asarray(data[s_col], dtype=np.float64),
                "actions": np.asarray(data[a_col], dtype=np.float64),
            }
            m_col = next((c for c in _MASK_CANDIDATES if c in keys), None)
            if m_col is not None:
                result["mask"] = np.asarray(data[m_col], dtype=bool)
            return result
    except Exception as e:
        logger.warning("npz 解析失败 %s: %s", path, e)
        return None


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    """按优先级尝试多种格式加载真实数据集。

    优先级：.npz 文件 > LeRobot 目录(data/*.parquet) > canonical episode 目录。
    返回 {"state": [N, Ds], "actions": [N, Da], "mask": [N] (可选)}。
    """
    if not path.exists():
        raise FileNotFoundError(f"dataset-path 不存在: {path}")

    # 1. .npz
    if path.is_file():
        data = _load_from_npz(path)
        if data is not None:
            logger.info("从 npz 加载: %s", path)
            return data
        raise ValueError(f"无法解析文件: {path}（支持 .npz，含 state/actions 字段）")

    # 2. LeRobot 目录
    data = _load_from_lerobot_dir(path)
    if data is not None:
        logger.info("从 LeRobot 目录加载: %s", path)
        return data

    # 3. canonical episode 目录
    data = _load_from_canonical_dir(path)
    if data is not None:
        logger.info("从 canonical episode 目录加载: %s", path)
        return data

    raise ValueError(
        f"无法识别的数据集格式: {path}\n"
        "支持：.npz 文件 / LeRobot 目录(data/*.parquet) / canonical episode 目录(子目录含 steps.parquet|steps.hdf5)"
    )


# ---------------------------------------------------------------------------
# 维度校验（红线：维度严格按 train_config.py）
# ---------------------------------------------------------------------------
def validate_dimensions(data: Dict[str, np.ndarray]) -> None:
    """校验 actions 末维 == ACTION_DIM（train_config.py）。state 维度由数据决定。"""
    actions = np.asarray(data["actions"])
    if actions.ndim < 2:
        actions = actions.reshape(-1, 1)
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"actions 维度 {actions.shape[1]} != train_config.py action_dim={ACTION_DIM}（方案书 §3.4）"
        )
    state = np.asarray(data["state"])
    if state.ndim < 2:
        state = state.reshape(-1, 1)
    logger.info("维度校验通过：state[D=%d] actions[D=%d]（action_dim 源 train_config.py）",
                state.shape[1], actions.shape[1])


# ---------------------------------------------------------------------------
# QA 打印（方案书 §3.3.1：每维分布 / 1%99% 分位 / 夹爪双峰）
# ---------------------------------------------------------------------------
def print_qa_report(stats_by_key: Dict[str, Dict[str, np.ndarray]], quiet: bool) -> None:
    """打印每维 mean/std/min/max/q01/q99，供 QA 检查分布与夹爪双峰。"""
    if quiet:
        return
    print("-" * 72)
    print("归一化统计量 QA 报告（方案书 §3.3.1：每维分布 / 1%99% 分位 / 夹爪双峰）")
    print("-" * 72)
    for key in NORM_STATS_KEYS:
        if key not in stats_by_key:
            continue
        s = stats_by_key[key]
        dim = s["mean"].shape[0]
        print(f"[{key}] dim={dim}")
        header = f"  {'dim':>4} | {'mean':>12} | {'std':>12} | {'min':>12} | {'max':>12} | {'q01':>12} | {'q99':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in range(dim):
            print(f"  {i:>4} | {s['mean'][i]:>12.6f} | {s['std'][i]:>12.6f} | "
                  f"{s['min'][i]:>12.6f} | {s['max'][i]:>12.6f} | "
                  f"{s['q01'][i]:>12.6f} | {s['q99'][i]:>12.6f}")
    print("-" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 π0.5 训练数据集计算归一化统计量（openpi NormStats JSON + SHA256）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="数据集路径（LeRobot 目录 / canonical episode 目录 / .npz 文件）。"
             "启用 --mock 时可选；未启用 --mock 时必填。",
    )
    parser.add_argument(
        "--output-path", default="./norm_stats.json",
        help="输出 JSON 路径（openpi NormStats 标准格式）。",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="生成合成随机数据测试（CPU 兼容，无 GPU/openpi 也能跑）。",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="静默模式：只输出结果与 SHA256，不打印进度 / QA 报告。",
    )
    args = parser.parse_args()

    # --mock 与 --dataset-path 互斥逻辑：未启用 --mock 时 --dataset-path 必填
    if not args.mock and not args.dataset_path:
        parser.error("未启用 --mock 时必须提供 --dataset-path")

    return args


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    # ---- 0. 从 train_config.py 读取 action_dim（红线：维度严格按 train_config）----
    # 延迟到此处（日志级别已按 --quiet 调整），避免静默模式泄漏进度信息。
    global ACTION_DIM
    ACTION_DIM = _load_action_dim_from_train_config()

    # ---- 1. 加载数据 ----
    if args.mock:
        logger.info("【Mock 模式】使用 numpy 合成随机数据（CPU 兼容验证）")
        data = generate_mock_data()
    else:
        assert args.dataset_path is not None  # parse_args 已保证
        data = load_dataset(Path(args.dataset_path))

    # ---- 2. 维度校验（红线：维度严格按 train_config.py）----
    validate_dimensions(data)

    mask = data.get("mask")
    if mask is None:
        logger.info("数据未含 mask 字段，按全有效处理（无 padding 过滤）")

    # ---- 3. 逐键计算统计量 ----
    stats_by_key: Dict[str, Dict[str, np.ndarray]] = {}
    for key in NORM_STATS_KEYS:
        if key not in data:
            logger.warning("数据中缺少键 %s，跳过", key)
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # mask 对所有键共用（按样本维过滤）
        stats_by_key[key] = compute_stats(arr, mask=mask, key=key)
        logger.info("[%s] 统计量计算完成: shape=%s mean[0]=%.6f std[0]=%.6f",
                    key, arr.shape, stats_by_key[key]["mean"][0], stats_by_key[key]["std"][0])

    if not stats_by_key:
        print("ERROR: 未计算出任何统计量（数据为空或字段缺失）")
        return 1

    # ---- 4. 构造 NormStats 并保存 JSON（Schema 100% 对齐 openpi）----
    norm_stats = build_norm_stats(stats_by_key)
    output_path = Path(args.output_path)
    save_norm_stats(output_path, norm_stats)
    logger.info("已保存 norm_stats JSON: %s", output_path)

    # ---- 5. SHA256 校验和（供 model_manifest.yaml，方案书 §8.5 / §7.2）----
    sha256_full = compute_sha256(output_path)
    sha256_short = sha256_full[:16]  # 与 src/executors/pi05.py 的 _norm_stats_sha 截断一致

    # ---- 6. 输出 ----
    if not args.quiet:
        print_qa_report(stats_by_key, quiet=False)
        print(f"openpi.shared.normalize 可用: {OPENPI_NORMALIZE_AVAILABLE}")
        print(f"action_dim (源 train_config.py): {ACTION_DIM}")
        print(f"输出文件: {output_path}")
        print(f"统计键: {list(stats_by_key.keys())}")

    print()
    print("=" * 72)
    print("归一化统计量计算完成")
    print("=" * 72)
    print(f"output_path      = {output_path}")
    print(f"sha256 (full)    = {sha256_full}")
    print(f"sha256 (前16位)  = {sha256_short}  (与 pi05.py norm_stats_sha 截断一致)")
    print("=" * 72)
    print("下一步：将该 SHA256 写入 model_manifest.yaml（方案书 §8.5）；")
    print("       训练时设置 PI05_NORM_STATS_PATH 指向此文件（src/executors/pi05.py 追溯）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
