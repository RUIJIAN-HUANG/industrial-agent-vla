"""scripts/pi05/convert_openpi.py

将角色C提供的 canonical 轨迹数据（HDF5/Parquet）转换成 openpi 要求的 LeRobot 数据集格式。

负责人：E（π0.5/openpi）

方案书出处：
- §5.4 convert_openpi：canonical → LeRobot dataset
  （openpi DataConfig 键、padding、resize-with-pad、norm stats assets）。
- §3.3.1 Para186：自定义数据推荐 LeRobot；训练前计算 norm stats；
  本项目自有统计，不沿用 OpenVLA。
- §5.1 Table 38 / Canonical Episode Schema：
  episode/{meta.json, steps.parquet|hdf5, front_rgb/, wrist_rgb/, result.json, sha256.txt}。
- §5.4 跨格式一致性测试：图像 checksum 一致、7 维物理动作误差 <1e-6、指令完全一致。

流程位置：
    角色C canonical 轨迹(HDF5/Parquet)
      → 本脚本转换为 LeRobot 数据集
      → 跑 compute_norm_stats 算归一化（后续单独执行）
      → 用此数据集 + norm_stats 跑 LoRA 微调（JAX 路径，Table 21 Row3）

参考官方：openpi examples/libero/convert_libero_data_to_lerobot.py。

用法：
    python scripts/pi05/convert_openpi.py --data_dir /path/to/canonical/data
    python scripts/pi05/convert_openpi.py --data_dir /path/to/canonical/data \\
        --output_repo_id your_team/industrial --push_to_hub
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger("convert_openpi")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][convert_openpi] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 依赖：LeRobot（try/except，缺失时打印安装提示）
# ---------------------------------------------------------------------------
LeRobotDataset: Any = None
LEROBOT_AVAILABLE: bool = False
LEROBOT_IMPORT_ERROR: Optional[str] = None

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    LEROBOT_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    LEROBOT_IMPORT_ERROR = str(_e)


# ---------------------------------------------------------------------------
# 依赖：图像 IO（PIL 优先，cv2 兜底）
# ---------------------------------------------------------------------------
try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE: bool = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cv2  # type: ignore
    CV2_AVAILABLE: bool = True
except Exception:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# 依赖：parquet / hdf5
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
# 常量
# ---------------------------------------------------------------------------
ACTION_DIM: int = 7                       # [dx,dy,dz,dax,day,daz,gripper]，方案书 §5.1
DEFAULT_FPS: int = 10                     # 方案书 §3.4 control_hz=10
DEFAULT_STATE_DIM: int = 8                # Franka 7-DOF + 1 gripper
DEFAULT_ROBOT_TYPE: str = "franka"
DEFAULT_REPO_ID: str = "your_team/industrial"
DEFAULT_IMAGE_HW: tuple = (256, 256)      # openpi LIBERO 默认图像尺寸


# ---------------------------------------------------------------------------
# 图像加载
# ---------------------------------------------------------------------------
def load_image_as_array(path: Path, size: tuple = DEFAULT_IMAGE_HW) -> Optional[np.ndarray]:
    """加载 jpg 为 RGB uint8 ndarray，resize 到指定尺寸；失败返回 None。

    方案书 §5.4：LeRobot feature shape 固定 (256,256,3)，需在存储前 resize。
    注意：推理时 openpi input_transform 仍会做 resize-with-pad，属于不同路径，
    本脚本的 resize 仅为对齐 LeRobot feature schema。
    """
    if PIL_AVAILABLE:
        try:
            img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
            return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        except Exception as e:
            logger.warning("PIL 读取失败 %s: %s", path, e)
            return None
    if CV2_AVAILABLE:
        try:
            arr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
            if arr is None:
                return None
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
            return np.ascontiguousarray(arr, dtype=np.uint8)
        except Exception as e:
            logger.warning("cv2 读取失败 %s: %s", path, e)
            return None
    logger.warning("PIL/cv2 均不可用，无法读取图像 %s", path)
    return None


# ---------------------------------------------------------------------------
# steps 读取（parquet 优先，hdf5 兜底）
# ---------------------------------------------------------------------------
def load_steps(episode_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """读取 steps.parquet 或 steps.hdf5，返回 dict：
        robot_state: float32[N, d]
        action:      float32[N, 7]
        done:        bool[N] 或 None
    缺失或解析失败时返回 None。
    """
    parquet_path = episode_dir / "steps.parquet"
    hdf5_path = episode_dir / "steps.hdf5"

    if parquet_path.exists():
        if not PANDAS_AVAILABLE:
            logger.warning("需要 pandas 读取 parquet，但未安装: %s", parquet_path)
            return None
        try:
            df = pd.read_parquet(parquet_path)
            robot_state = np.asarray(
                [np.asarray(r, dtype=np.float32) for r in df["robot_state"]], dtype=np.float32
            )
            action = np.asarray(
                [np.asarray(a, dtype=np.float32) for a in df["action"]], dtype=np.float32
            )
            done = (
                np.asarray(df["done"].tolist(), dtype=bool)
                if "done" in df.columns else None
            )
            return {"robot_state": robot_state, "action": action, "done": done}
        except Exception as e:
            logger.warning("parquet 解析失败 %s: %s", parquet_path, e)
            return None

    if hdf5_path.exists():
        if not H5PY_AVAILABLE:
            logger.warning("需要 h5py 读取 hdf5，但未安装: %s", hdf5_path)
            return None
        try:
            with h5py.File(hdf5_path, "r") as f:
                robot_state = np.asarray(f["robot_state"], dtype=np.float32)
                action = np.asarray(f["action"], dtype=np.float32)
                done = np.asarray(f["done"]) if "done" in f else None
            return {"robot_state": robot_state, "action": action, "done": done}
        except Exception as e:
            logger.warning("hdf5 解析失败 %s: %s", hdf5_path, e)
            return None

    logger.warning("episode 缺少 steps.parquet/steps.hdf5: %s", episode_dir)
    return None


# ---------------------------------------------------------------------------
# episode 枚举
# ---------------------------------------------------------------------------
def find_episodes(data_dir: Path) -> List[Path]:
    """枚举 data_dir 下所有 episode 文件夹（含 meta.json）。"""
    episodes: List[Path] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "meta.json").exists():
            episodes.append(p)
    return episodes


# ---------------------------------------------------------------------------
# 预扫描：确定 state_dim
# ---------------------------------------------------------------------------
def detect_state_dim(data_dir: Path, fallback: int) -> int:
    """扫描第一个可读取的 episode 的第一个 step，获取 robot_state 维度。

    全部失败时返回 fallback。方案书 §5.1：robot_state[d]，d 由数据决定。
    """
    for ep_dir in find_episodes(data_dir):
        steps = load_steps(ep_dir)
        if steps is None:
            continue
        rs = steps["robot_state"]
        if rs.ndim == 2 and rs.shape[1] > 0:
            return int(rs.shape[1])
        if rs.ndim == 1 and rs.shape[0] > 0:
            # 整个 episode 是一维（单步），按 reshape 后推断
            return int(rs.shape[0])
    logger.warning("预扫描未找到有效 robot_state，使用 fallback state_dim=%d", fallback)
    return fallback


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将角色C canonical 轨迹数据转换为 LeRobot 数据集（openpi 路径）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="角色C canonical 数据根目录（每个子文件夹是一个 episode）。",
    )
    parser.add_argument(
        "--output_repo_id", default=DEFAULT_REPO_ID,
        help="输出 LeRobot 数据集名称（也作为本地缓存目录名）。",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="可选，转换完成后推送到 HuggingFace Hub。",
    )
    parser.add_argument(
        "--fps", type=int, default=DEFAULT_FPS,
        help="数据集 fps（方案书 §3.4 control_hz=10）。",
    )
    parser.add_argument(
        "--state_dim", type=int, default=DEFAULT_STATE_DIM,
        help="机器人本体状态维度（Franka 7-DOF+gripper=8）。传 0 表示自动检测。",
    )
    parser.add_argument(
        "--robot_type", default=DEFAULT_ROBOT_TYPE,
        help="LeRobot robot_type 标识。",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=list(DEFAULT_IMAGE_HW),
        help="图像 resize 尺寸（H W），默认 256 256。",
    )
    parser.add_argument(
        "--filter_success_only", action="store_true",
        help="仅转换 result.json 中 success=true 的 episode（方案书 §5.3：标准成功占约 70%）。",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    # ---- LeRobot 依赖检查 ----
    if not LEROBOT_AVAILABLE:
        print("ERROR: lerobot 未安装或 import 失败。")
        if LEROBOT_IMPORT_ERROR:
            print(f"  原因: {LEROBOT_IMPORT_ERROR}")
        print("  请安装: pip install lerobot")
        print("  或参考 openpi 文档: https://github.com/Physical-Intelligence/openpi")
        return 1

    # ---- 图像 IO 检查 ----
    if not PIL_AVAILABLE and not CV2_AVAILABLE:
        print("ERROR: 图像 IO 依赖缺失（PIL 和 cv2 都不可用）。")
        print("  请安装: pip install pillow  或  pip install opencv-python")
        return 1

    # ---- 输入检查 ----
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data_dir 不存在或不是目录: {data_dir}")
        return 1

    episodes = find_episodes(data_dir)
    if not episodes:
        print(f"ERROR: data_dir 下未找到任何 episode（需含 meta.json 的子目录）: {data_dir}")
        return 1

    # ---- 确定 state_dim ----
    if args.state_dim and args.state_dim > 0:
        state_dim: int = args.state_dim
        logger.info("使用指定 state_dim=%d", state_dim)
    else:
        state_dim = detect_state_dim(data_dir, fallback=DEFAULT_STATE_DIM)
        logger.info("自动检测 state_dim=%d", state_dim)

    img_h, img_w = int(args.image_size[0]), int(args.image_size[1])
    img_shape: tuple = (img_h, img_w, 3)

    # ---- 创建 LeRobot 数据集 ----
    features: Dict[str, Any] = {
        "image": {
            "dtype": "image",
            "shape": img_shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": img_shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
    }

    logger.info(
        "创建 LeRobot 数据集: repo_id=%s fps=%d robot_type=%s state_dim=%d image=%s",
        args.output_repo_id, args.fps, args.robot_type, state_dim, img_shape,
    )
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.output_repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            features=features,
        )
    except Exception as e:
        print(f"ERROR: LeRobotDataset.create 失败: {e}")
        return 1

    # ---- 统计容器 ----
    stats: Dict[str, Any] = {
        "total_episodes": 0,        # 成功写入的 episode 数
        "total_steps": 0,           # 成功写入的 step 数
        "skipped_steps": 0,         # 跳过的 step 数
        "skipped_episodes": 0,      # 整个 episode 跳过的数量
        "skipped_reasons": {},      # 跳过原因 → 计数
    }

    def record_skip(reason: str) -> None:
        stats["skipped_steps"] += 1
        stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1

    # ---- 遍历 episode ----
    for ep_dir in episodes:
        ep_name = ep_dir.name

        # 1. 读取 meta.json
        meta_path = ep_dir / "meta.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning("[%s] meta.json 读取失败，跳过 episode: %s", ep_name, e)
            stats["skipped_episodes"] += 1
            continue

        instruction = str(meta.get("instruction", "")).strip()
        if not instruction:
            logger.warning("[%s] meta.json 缺少 instruction，使用空字符串", ep_name)

        # 2. 可选：按 result.json 过滤失败 episode（方案书 §5.1 / §5.3）
        if args.filter_success_only:
            result_path = ep_dir / "result.json"
            if result_path.exists():
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    if not result.get("success", True):
                        logger.info("[%s] result.json success=false，跳过（--filter_success_only）", ep_name)
                        stats["skipped_episodes"] += 1
                        stats["skipped_reasons"]["result_not_success"] = (
                            stats["skipped_reasons"].get("result_not_success", 0) + 1
                        )
                        continue
                except Exception as e:
                    logger.warning("[%s] result.json 读取失败：%s，不跳过", ep_name, e)
            else:
                logger.warning(
                    "[%s] --filter_success_only 启用但 result.json 缺失，不跳过（按通过处理）",
                    ep_name,
                )

        # 3. 读取 steps
        steps = load_steps(ep_dir)
        if steps is None:
            logger.warning("[%s] steps 读取失败，跳过 episode", ep_name)
            stats["skipped_episodes"] += 1
            continue

        robot_states = steps["robot_state"]  # [N, d] 或 [N*d]
        actions = steps["action"]            # [N, 7] 或 [N*7]
        n_steps = int(len(robot_states))
        if n_steps == 0:
            logger.warning("[%s] steps 为空，跳过 episode", ep_name)
            stats["skipped_episodes"] += 1
            continue

        # 规整成 2D
        if robot_states.ndim == 1:
            robot_states = robot_states.reshape(n_steps, -1)
        if actions.ndim == 1:
            actions = actions.reshape(n_steps, -1)

        front_dir = ep_dir / "front_rgb"
        wrist_dir = ep_dir / "wrist_rgb"
        has_wrist_dir = wrist_dir.exists()

        # 4. 逐 step 处理
        added = 0
        for i in range(n_steps):
            action = actions[i]
            state = robot_states[i]

            # ---- 数据清洗（方案书 §5.1 静默错误防护） ----
            # (1) action 维度必须为 7
            if action.shape[0] != ACTION_DIM:
                logger.warning(
                    "[%s step=%d] action 维度=%d 不等于 %d，跳过",
                    ep_name, i, int(action.shape[0]), ACTION_DIM,
                )
                record_skip("action_dim_mismatch")
                continue
            # (2) action NaN/Inf
            if not np.all(np.isfinite(action)):
                logger.warning("[%s step=%d] action 含 NaN/Inf，跳过", ep_name, i)
                record_skip("action_nan_inf")
                continue
            # (3) state NaN/Inf
            if not np.all(np.isfinite(state)):
                logger.warning("[%s step=%d] robot_state 含 NaN/Inf，跳过", ep_name, i)
                record_skip("state_nan_inf")
                continue
            # (4) state 维度一致
            if state.shape[0] != state_dim:
                logger.warning(
                    "[%s step=%d] robot_state 维度=%d 不等于期望 %d，跳过",
                    ep_name, i, int(state.shape[0]), state_dim,
                )
                record_skip("state_dim_mismatch")
                continue
            # (5) 前视图文件存在
            img_path = front_dir / f"{i:06d}.jpg"
            if not img_path.exists():
                logger.warning("[%s step=%d] 前视图不存在: %s，跳过", ep_name, i, img_path)
                record_skip("front_image_missing")
                continue
            image = load_image_as_array(img_path, size=(img_h, img_w))
            if image is None:
                logger.warning("[%s step=%d] 前视图读取失败: %s，跳过", ep_name, i, img_path)
                record_skip("front_image_load_failed")
                continue

            # (6) 腕部视图（可选；缺失用黑图占位，不跳过）
            wrist_image: Optional[np.ndarray] = None
            if has_wrist_dir:
                wrist_path = wrist_dir / f"{i:06d}.jpg"
                if wrist_path.exists():
                    wrist_image = load_image_as_array(wrist_path, size=(img_h, img_w))
                if wrist_image is None:
                    logger.warning(
                        "[%s step=%d] 腕部图缺失或读取失败，使用黑图占位", ep_name, i
                    )
            if wrist_image is None:
                wrist_image = np.zeros(img_shape, dtype=np.uint8)

            # ---- 添加帧 ----
            try:
                dataset.add_frame({
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": np.asarray(state, dtype=np.float32),
                    "actions": np.asarray(action, dtype=np.float32),
                })
                added += 1
            except Exception as e:
                logger.warning("[%s step=%d] add_frame 失败: %s，跳过", ep_name, i, e)
                record_skip("add_frame_failed")
                continue

        # 5. 保存 episode
        if added > 0:
            try:
                dataset.save_episode(task=instruction)
                stats["total_episodes"] += 1
                stats["total_steps"] += added
                logger.info(
                    "[%s] 写入完成: %d steps, task=%r",
                    ep_name, added, instruction[:60],
                )
            except Exception as e:
                logger.error("[%s] save_episode 失败: %s", ep_name, e)
                stats["skipped_episodes"] += 1
        else:
            logger.warning("[%s] 全部 step 被跳过，未调用 save_episode", ep_name)
            stats["skipped_episodes"] += 1

    # ---- consolidate（不计算 norm stats，后续单独算）----
    # 方案书 §3.3.1 Para186：norm_stats 用 compute_norm_stats 单独生成本项目自有统计。
    logger.info("consolidate 数据集（run_compute_stats=False，norm stats 后续单独算）")
    try:
        dataset.consolidate(run_compute_stats=False)
    except Exception as e:
        logger.error("consolidate 失败: %s", e)
        # 继续打印统计

    # ---- push to hub（可选）----
    if args.push_to_hub:
        try:
            logger.info("推送到 HuggingFace Hub: %s", args.output_repo_id)
            dataset.push_to_hub()
        except Exception as e:
            logger.warning("push_to_hub 失败: %s", e)

    # ---- 推断输出本地路径（best-effort）----
    output_path_str: str = ""
    for attr in ("root_path", "data_dir", "path"):
        v = getattr(dataset, attr, None)
        if v is not None:
            output_path_str = str(v)
            break
    if not output_path_str:
        # 兜底：HuggingFace lerobot 缓存
        output_path_str = f"~/.cache/huggingface/lerobot/{args.output_repo_id}"

    # ---- 打印转换统计 ----
    print()
    print("=" * 64)
    print("=== 转换统计 ===")
    print("=" * 64)
    print(f"总 episode 数（成功写入）: {stats['total_episodes']}")
    print(f"跳过 episode 数:           {stats['skipped_episodes']}")
    print(f"总 step 数（成功写入）:    {stats['total_steps']}")
    print(f"跳过 step 数:              {stats['skipped_steps']}")
    if stats["skipped_reasons"]:
        print("跳过原因明细:")
        for reason, count in sorted(stats["skipped_reasons"].items()):
            print(f"  {reason}: {count}")
    print(f"输出数据集 repo_id: {args.output_repo_id}")
    print(f"本地路径:           {output_path_str}")
    print("=" * 64)
    print("下一步：")
    print("  1. 跑 compute_norm_stats 生成归一化统计（方案书 §3.3.1 Para186）")
    print("  2. 用此数据集 + norm_stats 跑 LoRA 微调（方案书 Table 21 Row3，JAX 路径）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
