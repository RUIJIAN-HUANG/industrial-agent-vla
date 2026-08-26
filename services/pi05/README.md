# π0.5 / openpi Service

> 场景口径（2026-08-18）：下述“抓取四个零件、2×3 料箱”是 V1
> `single_bin_pack_handoff_v1` 的冻结 TaskProfile。当前 V2 训练域是 8 工件、2×4
> 料箱的人工工业采集场景；V2 Canonical 数据合同和 N−9 转换 Gate 已实现，但 B
> 侧 GUI/物理采集入口仍缺失，尚未宣称 π0.5 已自动控制该场景。

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

## Canonical V2 数据 Gate

V2 固定 `canonical_schema_version=2.0`、场景
`single_bin_manual_industrial_v2`、任务 `P01_TO_S11` 和训练指令
`请将螺母 P01 放置到料箱的 S11 格子中。`。`scripts/pi05/canonical_v2.py` 在读取时复核 JSON Schema、
HDF5 SHA、三相机/双臂 Stream、有限 `float32[N,7]` state/action、无 padding 和
Arm_A/`pi05` 身份。

不依赖 LeRobot 的只读 Preflight：

```powershell
python scripts\pi05\convert_openpi_v2.py `
  --data-dir <CANONICAL_V2_ROOT> `
  --split-registry <SPLIT_REGISTRY_JSON> `
  --preflight-only
```

真实转换在固定训练环境安装 LeRobot 后执行：

```powershell
python scripts\pi05\convert_openpi_v2.py `
  --data-dir <CANONICAL_V2_ROOT> `
  --split-registry <SPLIT_REGISTRY_JSON> `
  --output-dir <LEROBOT_DATASET_ROOT> `
  --repo-id <ORG/REPO_ID>
```

动作流必须连续 10 Hz；N 条动作严格生成 N−9 个 `[10,7]` 窗口。N<10、缺失
精确 tick 观测、padding 或任何动作数值变化都会阻止发布。

仓库已提供 [`handler.py`](handler.py) 的 `build_v1_infer_handler()` 作为
`POST /v1/infer` 强制入口核心：它先解析并校验 CAS，再把只读 RGB 数组替换进
`model_input` 后调用注入的 π0.5 backend。HTTP/WebSocket 外壳不得绕过该 handler。

## Canonical v1 数据 Gate

`scripts/pi05/canonical_v1.py` 是角色 E 的薄适配层，底层强制复用主线
`industrial_agent.data.CanonicalEpisodeReader`。只接受权威
`episode.h5 + structure.json` Canonical Episode 和经过 SHA 校验的外部 Split
Registry；不得复制 Reader 或另建 Canonical 格式。旧的
`meta.json + steps.jsonl`、`steps.parquet`、`steps.hdf5` 和 `front_rgb` 均不接受。

每个 `valid_mask=true` 的 Arm_A/`pi05` 动作以自身 `physics_tick` 为锚点，必须
精确找到同 tick 的 Arm_A 状态和 `CAM_A_TOP` 帧；缺失、fallback、Arm_B 混入、
7D 维度错误、NaN/Inf、时间戳/sequence_id/SHA/Split 不合法均 fail-closed。
Episode 指令必须与 `single_bin_pack_handoff_v1` 的 Arm_A 冻结原文逐字一致，动作
必须属于 `S01_ARM_A_PACK_HANDOFF`；非 `SUCCEEDED` Episode 保留作 QA 证据，但
不得进入 LeRobot 模仿数据或 norm stats。
不同频率流不要求逐行同 tick。转换阶段保留原始 1280×720 RGB，不执行
224×224 缩放。

生产 state/action 已冻结为
`[x_m,y_m,z_m,ax_rad,ay_rad,az_rad,gripper_norm]`，适配层直接使用权威 Reader
验证后的 7D rotvec 数据，不重建另一套姿态表示。生产仍要求通过
`--state-mapper` 显式注入
`scripts.pi05.canonical_v1:CanonicalPi05StateMapper`，没有隐式默认值。

固定 LeRobot API 没有 `consolidate()`。发布 Gate 是：保存 Episode、关闭 writer、
从 staging 离线重开、Loader 全量遍历和 provenance/SHA 校验，全部成功后才原子
发布。`pi05_provenance.sha256` 是离线 Loader 的必需篡改检测文件。

Canonical→LeRobot（`10` 是已冻结模型采样 FPS，不是 action horizon）：

```bash
python scripts/pi05/convert_openpi.py \
  --data_dir <CANONICAL_ROOT> \
  --split-registry <SPLIT_REGISTRY_JSON> \
  --project-root <PROJECT_GIT_ROOT> \
  --openpi-root <CLEAN_OPENPI_GIT_ROOT> \
  --openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
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
  --project-root <PROJECT_GIT_ROOT> \
  --openpi-root <CLEAN_OPENPI_GIT_ROOT> \
  --openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --manifest <LEROBOT_DATASET_ROOT>/pi05_provenance.json
```

Train-only norm-stats：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/pi05/compute_norm_stats.py \
  --dataset-path <LEROBOT_DATASET_ROOT> \
  --split-registry <SPLIT_REGISTRY_JSON> \
  --project-root <PROJECT_GIT_ROOT> \
  --openpi-root <CLEAN_OPENPI_GIT_ROOT> \
  --openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --input-format lerobot \
  --state-mapper scripts.pi05.canonical_v1:CanonicalPi05StateMapper \
  --repo-id <ORG/REPO_ID> \
  --manifest <LEROBOT_DATASET_ROOT>/pi05_provenance.json \
  --output-path <OUTPUT_DIR>/norm_stats.json
```

转换阶段保留原始 uint8 1280×720 RGB；resize-with-pad 属于 OpenPI transform。
### 合并 Gate 与发布 Gate

Issue #15 的合并范围是 synthetic Canonical→LeRobot Gate，不包含 Linux Isaac Sim
或真实 Episode。合并前运行：

```bash
python -m pytest tests/pi05 -q
python -m ruff check configs/pi05 scripts/pi05 tests/pi05
python -m ruff format --check configs/pi05 scripts/pi05 tests/pi05
git diff --check
```

上述 synthetic Gate 全部通过、且 action horizon 仍保持显式未冻结哨兵时，可以按
Issue #15 合并；不得把 synthetic PASS 描述成真实 Isaac 数据或完整训练验收。

后续正式发布 Gate 必须使用 5 条真实 Arm_A Isaac
Golden Episode，完成关闭/重开/全量遍历并至少抽查 10 条来源映射。正式
norm-stats 只能读取同一已验证 Split Registry 的 Train Split，并必须使用真实
`openpi.shared.normalize`；Loader 会按每个 `canonical_episode_id` 重新查询 Registry，
不信任 sidecar 自报的 split。`action_horizon=10` 是
ARCH-2026-001 item 4 的未冻结 LIBERO 哨兵，正式训练配置会直接报错，不会静默
使用。真实 LeRobot/OpenPI 集成测试由 Ubuntu/Docker 环境执行。

正式放行必须指定一个不存在的新证据目录；Gate 会持久化 LeRobot 数据、
`norm_stats.json`、来源 manifest 和 `pi05-release-gate.json` PASS 报告：

```bash
PI05_REAL_CANONICAL_ROOT=<FIVE_REAL_EPISODES_ROOT> \
PI05_REAL_SPLIT_REGISTRY=<SPLIT_REGISTRY_JSON> \
PI05_PROJECT_ROOT=<PROJECT_GIT_ROOT> \
PI05_OPENPI_ROOT=<CLEAN_PINNED_OPENPI_GIT_ROOT> \
PI05_RELEASE_OUTPUT_ROOT=<NEW_RELEASE_EVIDENCE_DIR> \
pytest -q tests/pi05/test_openpi_data_pipeline.py \
  -k real_five_episode_lerobot_openpi_release_gate
```

只有测试返回 PASS、报告中的 Episode 数为 5、roundtrip 抽查数至少为 10，且
所有产物 SHA 可复核时，才能勾选 PR 的真实数据、离线 Loader 和 norm-stats Gate。

转换、离线 Loader 和 norm-stats 必须使用同一个 provenance producer：当前项目
完整 Git SHA、工作树 dirty 标志、包含未跟踪文件内容的工作树 diff SHA-256，以及
从 `--openpi-root` 实际解析的冻结 OpenPI Commit。OpenPI checkout 必须洁净且 HEAD
必须等于固定 Commit；项目开发阶段允许 dirty 工作树，但正式发布必须保存完整证据；
禁止使用 `unknown` 或空值代替来源。

不要在此目录提交 checkpoint、训练数据、缓存或个人机器路径。完整接口见
[`../../docs/architecture/interface-contracts.md`](../../docs/architecture/interface-contracts.md)。
