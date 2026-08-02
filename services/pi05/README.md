# π0.5 / openpi Service

负责人：E。当前状态：接口占位，真实模型尚未集成。

冻结定位：π0.5 是 Arm_A 的唯一 VLA。它直接接收预设的上游自然语言、
`CAM_A_TOP` 完整图像和 Arm_A 状态，负责抓取四个零件、纠正倒放零件、
装入 2×3 料箱、把满箱放到 `HANDOFF_CENTER` 并退回 `HOME_A`。
YOLO DetectionPacket 不是推理前置条件。π0.5 必须针对这一固定角色完成工业
场景微调并提供 base/tuned 同协议对照。

本目录只放 π0.5/openpi 独立服务的生产代码、依赖、示例配置和测试。实现前必须
固定并记录：

- 上游仓库 Commit、checkpoint SHA-256、norm stats SHA-256；
- LeRobot 数据映射、相机顺序、语言字段和任务 ID；
- JAX 推理输出到统一 `N×7` 动作的转换；
- `/health`、`/v1/infer`、`/v1/cancel` 的超时、错误码和幂等语义；
- 与 `schemas/executor-*.schema.json`、`action-chunk.schema.json` 的契约测试。
- 工业微调的数据/配置/checkpoint SHA、base/tuned 成功率与失败分布。
- 服务只能输出 `arm_id=Arm_A` 的动作；收到 Arm_B 请求必须拒绝。
- 恢复时必须使用 Arm_A 的新鲜观测重新推理，禁止请求 OpenVLA 接管。
- 服务入口必须调用
  `industrial_agent.service_images.CasRequestImageResolver.resolve_vla_request()`
  将 `CAM_A_TOP` 引用解析为真实 RGB；冻结场景的 `wrist_image` 必须为
  `null`。Real 模式缺图、坏 SHA 或解码失败时必须 fail-closed，禁止使用零图、
  placeholder 或自动降级 Mock。

仓库已提供 [`handler.py`](handler.py) 的 `build_v1_infer_handler()` 作为
`POST /v1/infer` 强制入口核心：它先解析并校验 CAS，再把只读 RGB 数组替换进
`model_input` 后调用注入的 π0.5 backend。HTTP/WebSocket 外壳不得绕过该 handler。

## Canonical v1 数据 Gate

`scripts/pi05/canonical_v1.py` 是转换与 norm-stats 共用的唯一读取器，只接受
`meta.json + steps.jsonl + rgb/CAM_A_TOP + checksums.sha256`。旧的
`steps.parquet`、`steps.hdf5` 和 `front_rgb` 会直接失败。转换命令必须显式提供
正整数 `--fps`、非负整数 `--timestamp-tolerance-ns` 和 `module:attribute` 形式
的 `--state-mapper`；时间戳间隔必须在显式容差内匹配 FPS。转换阶段保留原始
1280×720 RGB，不执行 224×224 缩放。

生产 state 已冻结为
`[x_m,y_m,z_m,ax_rad,ay_rad,az_rad,gripper_norm]`。Canonical TCP 固定使用
`[x,y,z,qx,qy,qz,qw]`（xyzw 不可配置），并转换为最短 rotation-vector；最后一维
只由 `robot.arm_a.gripper_open` 布尔值映射为 1.0/0.0，不读取
`gripper_state` 阈值。生产仍要求通过 `--state-mapper` 显式注入
`scripts.pi05.canonical_v1:CanonicalPi05StateMapper`，没有隐式默认值。

固定 LeRobot API 没有 `consolidate()`。发布 Gate 是：保存 Episode、关闭 writer、
从 staging 离线重开、Loader 全量遍历和 provenance/SHA 校验，全部成功后才原子
发布。`pi05_provenance.sha256` 是离线 Loader 的必需篡改检测文件。

Canonical→LeRobot（`10` 是已冻结模型采样 FPS，不是 action horizon）：

```bash
python scripts/pi05/convert_openpi.py \
  --data_dir <CANONICAL_ROOT> \
  --output-dir <LEROBOT_DATASET_ROOT> \
  --output_repo_id <ORG/REPO_ID> \
  --fps 10 \
  --timestamp-tolerance-ns <APPROVED_TOLERANCE_NS> \
  --state-mapper scripts.pi05.canonical_v1:CanonicalPi05StateMapper
```

离线 Loader：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/pi05/smoke_lerobot_loader.py \
  --dataset-root <LEROBOT_DATASET_ROOT> \
  --repo-id <ORG/REPO_ID> \
  --manifest <LEROBOT_DATASET_ROOT>/pi05_provenance.json
```

Train-only norm-stats：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/pi05/compute_norm_stats.py \
  --dataset-path <LEROBOT_DATASET_ROOT> \
  --input-format lerobot \
  --state-mapper scripts.pi05.canonical_v1:CanonicalPi05StateMapper \
  --repo-id <ORG/REPO_ID> \
  --manifest <LEROBOT_DATASET_ROOT>/pi05_provenance.json \
  --output-path <OUTPUT_DIR>/norm_stats.json
```

转换阶段保留原始 uint8 1280×720 RGB；resize-with-pad 属于 OpenPI transform。
正式 norm-stats 必须使用真实 `openpi.shared.normalize`。`action_horizon=10` 是
ARCH-2026-001 item 4 的未冻结 LIBERO 哨兵，正式训练配置会直接报错，不会静默
使用。真实 LeRobot/OpenPI 集成测试由 Ubuntu/Docker 环境执行。

不要在此目录提交 checkpoint、训练数据、缓存或个人机器路径。完整接口见
[`../../docs/architecture/interface-contracts.md`](../../docs/architecture/interface-contracts.md)。
