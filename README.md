# XH-202607 工业环境 VLA 智能体

面向工业环境中的视觉感知、语言指令和机械臂执行。本仓库目前以 V2
`single_bin_manual_industrial_v2` 为唯一正式开发口径，重点维护可审计的
π0.5/Arm_A 连续闭环、人工 Canonical Episode 采集和模型服务合同。

> 当前边界（2026-08-29）：V2 场景、任务合同、Canonical V2 Recorder/Reader、
> Pi0.5 数据 Preflight 和相关 CI 已进入 `main`。完整的 Isaac Sim GUI、物理、IK、
> 抓取、碰撞和满载搬运证据仍需在目标 Isaac Sim 环境中单独完成；代码存在不等于
> 真实机器人或正式比赛验收已经通过。

## 1. 当前正式闭环

```text
用户选择冻结指令
        ↓
总控校验 task_id、指令、对象和槽位的一一对应
        ↓
π0.5（唯一正式 VLA，固定控制 Arm_A）
        ↓
执行一个 7D 微动作 → 获取新鲜观测 → 再推理
        ↓
3 个新鲜终局帧中至少 2 票通过
        ↓
成功结束或安全停止
```

系统不设置额外的 NLP Agent，不在运行时改写用户指令，不让 YOLO DetectionPacket
成为 π0.5 推理前置条件。总控、YOLO 和单一 π0.5 是唯一运行时 Agent；π0.5
通过 `arm_id` 服务两只机械臂。

### 正式任务与精确指令

`task_id` 和用户指令必须逐字匹配。以下两条同时用于界面、采集数据、训练
和 Pi0.5 推理：

| task_id | 精确指令 | 目标 |
|---|---|---|
| `P01_TO_S11` | `把P01放到S11中` | 将轴件 P01 放入料箱 S11 |
| `W01_TO_S14` | `把W01放到S14中` | 将扳手 W01 放入料箱 S14 |

对应机器真源：[`configs/v2-task-profile.json`](configs/v2-task-profile.json)、
[`configs/mvp-instruction-options.json`](configs/mvp-instruction-options.json)。
不要把自然语言扩写成“请将……放置……”；任何标点、空格或措辞差异都可能导致
任务解析、Canonical 校验或 Pi0.5 服务拒绝请求。

当前 UI 还登记了三条未开放正式数据采集的任务：
`P03_UPRIGHT_TO_S12`、`BIN01_TO_FINISHED01` 和 `PACK_ALL_AND_FINISH`。
它们必须先完成各自的任务合同和验收，不能伪装成上述两个正式任务的数据。

## 2. V2 场景

- 两台 Franka：`Arm_A` 负责单件装箱，`Arm_B` 当前正式任务保持静止；
- 三台固定 RGB 相机：`CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP`；
- 8 个工业零件：P01–P04、N01–N02、W01–W02；
- 一个 `2×4` 料箱，槽位为 S11–S24，并使用中央提梁 `BIN_CARRY_TCP`；
- A/B/C/D 四个区域各放置 2 件物体，P03/P04 初始为倒立状态；
- 采集动作频率为 10 Hz，Canonical state/action 均为有限值的 `float32[N,7]`；
- 在线 Observation、VLA 输入和在线终局证据禁止包含 GT，GT 只能写入离线证据目录。

固定槽位映射：

| 槽位 | S11 | S12 | S13 | S14 | S21 | S22 | S23 | S24 |
|---|---|---|---|---|---|---|---|---|
| 零件 | P01 | P03 | N01 | W01 | P02 | P04 | N02 | W02 |
| 类型 | 轴件 | 轴件 | 螺母 | 扳手 | 轴件 | 轴件 | 螺母 | 扳手 |

场景机器真源是 [`simulation/configs/single_bin_scene_v2.json`](simulation/configs/single_bin_scene_v2.json)，
完整采集顺序见 [`docs/v2-manual-industrial-collection.md`](docs/v2-manual-industrial-collection.md)。

## 3. 快速开始

要求 Python 3.10+。普通 Python 环境可以运行合同检查、Mock 回归和大部分单元测试；
Isaac Sim、LeRobot、OpenPI 和真实模型权重需要各自固定的专用环境。

### 安装测试依赖

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

macOS/Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

### 本地质量检查

```powershell
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python scripts/check_repository_hygiene.py
git diff --check
```

两份官方 PDF、冻结架构图和初版方案快照位于 [`docs/official/`](docs/official/)
及 [`docs/assets/`](docs/assets/)。任何 PR 都不得修改官方 PDF，或弱化每步重观察、
失败恢复、仿真初赛和安全停止等硬要求。

## 4. V2 仿真与人工采集

### 静态场景合同

```powershell
python simulation\run_v2_scene_acceptance.py `
  --evidence-dir artifacts\v2\static
```

这一步只验证配置、资产、槽位、质量预算和相机合同，不能替代 GUI、HOME、IK、
碰撞、抓取或满载搬运证据。

### Isaac Sim 验收顺序

按以下顺序执行，并为每一步保存证据：

1. 可见 GUI 构建场景、保存 USD、检查三路相机图；
2. 验证两臂 HOME、目标位 IK、碰撞和共享区安全互锁；
3. 验证抓取、放置和满载搬运；
4. 通过采集预检后，使用键盘入口采集正式 Canonical Episode。

主要入口：

| 入口 | 用途 |
|---|---|
| `simulation/run_v2_gui_scene_acceptance.py` | GUI 场景与相机证据 |
| `simulation/run_v2_home_acceptance.py` | 两臂 HOME 验收 |
| `simulation/run_v2_ik_reachability_acceptance.py` | V2 目标位 IK 验收 |
| `simulation/run_v2_dual_arm_micro_motion_acceptance.py` | 双臂微动作与安全门禁 |
| `simulation/run_v2_keyboard_collection.py` | 人工键盘 Canonical Episode 采集 |
| `simulation/v2_collection_preflight.py` | 正式采集前检查 |

### Canonical V2 → LeRobot / Pi0.5

拿到成功母 Episode 和经过 SHA 校验的 Split Registry 后，先运行只读 Preflight：

```powershell
python scripts\pi05\convert_openpi_v2.py `
  --data-dir <CANONICAL_V2_ROOT> `
  --split-registry <SPLIT_REGISTRY_JSON> `
  --preflight-only
```

V2 转换要求连续 10 Hz 动作；N 条动作只能生成 N−9 个完整 `[10,7]` 窗口。
缺少精确 tick、padding、NaN/Inf、错误 task identity 或非 Arm_A/`pi05` 身份时，
流程必须 fail-closed。正式训练还需要在固定 LeRobot/OpenPI 环境中执行转换、
norm stats 和 release gate，详见 [`services/pi05/README.md`](services/pi05/README.md)。

## 5. Pi0.5 服务边界

π0.5 服务使用 V2 配置和固定角色：

- 服务启动必须设置 `PI05_TASK_PROFILE_VERSION=v2`；
- 只接受 `P01_TO_S11` 或 `W01_TO_S14` 的精确指令；
- 只向 `Arm_A` 输出统一的 7D 动作；
- 请求入口必须先通过 CAS 图像解析和输入合同校验；
- `CAM_A_TOP` 是 V2 Pi0.5 的完整 RGB 输入，`wrist_image` 必须为 `null`；
- 缺图、坏 SHA、解码失败、错误机械臂或错误指令都必须拒绝请求；
- 生产配置中的 checkpoint 和 norm-stats 必须替换为完整的 `sha256:<64位摘要>`，
  不能使用 `latest`、占位符或版本昵称。

默认配置：

- [`configs/agent.default.json`](configs/agent.default.json)
- [`configs/agent.v2.default.json`](configs/agent.v2.default.json)
- [`configs/v2-task-profile.json`](configs/v2-task-profile.json)
- 服务实现：[`services/pi05/`](services/pi05/)

### 当前提交边界

π0.5 的工业策略尚未完成训练和最终验收。因此当前仓库是“工程提交候选版”，
不是可直接部署的模型发布版：

- 任务合同、Supervisor、7D 动作安全边界、数据 Recorder/Reader、转换 Preflight
  和服务接口可以进行审计与复现；
- `configs/agent.default.json` 中的 `checkpoint_sha` 和 `norm_stats_sha` 仍是占位符，
  这是有意保留的 fail-closed 状态，服务不会用占位符启动生产推理；
- `reports/evidence-index.md` 中真实 VLA 闭环证据仍为 `PENDING`，不能用 Mock、
  静态检查或接口测试替代真实模型结果；
- 训练完成后必须补充外部 checkpoint、norm stats、完整 SHA-256、训练环境和评测
  报告，再把模型清单状态从 `TRAINING` 更新为 `CANDIDATE` 或 `FROZEN`。

## 6. 当前模型与制品溯源

仓库不提交 `.pt`、`.ckpt`、`.pth`、`.safetensors` 或 `.onnx` 权重，只提交模型卡、
来源、兼容性和固定 SHA。当前 Manual-994 YOLO 候选的元数据已合入 `main`：

| 项目 | 当前值 |
|---|---|
| 模型 | `yolo_manual994_yolo11n_e10_cpu` |
| 权重文件 | `manual994/best.pt` |
| 权重 SHA-256 | `sha256:67a70dd1f575919bde9184a993097771bbdbaa7516cdd251c1f91b2a490f1e5c` |
| 来源仓库 | [`industrial-agent-vla-model-yolo-manual800`](https://github.com/RUIJIAN-HUANG/industrial-agent-vla-model-yolo-manual800) |
| 来源 commit | `7e4c37ad01831e08d87239a26cfed65f8b3b8d99` |
| 训练数据 | 994 张人工清洗图像，train/val/test = 810/105/79 |
| held-out mAP50 / mAP50-95 | `0.936 / 0.793` |

Pi0.5 当前尚未有可发布 checkpoint；YOLO Manual-994 也仍是感知候选模型，不能据此
宣称完整 VLA 闭环或生产放行。完整信息见 [`models/MODEL_CARD_yolo_manual994.md`](models/MODEL_CARD_yolo_manual994.md)、
[`models/CHECKSUMS_yolo_manual994.json`](models/CHECKSUMS_yolo_manual994.json) 和
[`models/MANIFEST.md`](models/MANIFEST.md)。模型元数据进入 Git 不代表权重已经下载、
真实三相机探针或生产门禁已经通过。

## 7. 目录导航

```text
configs/       运行配置、任务目录和服务配置
data/          数据卡、Manifest 和小型测试夹具；不放训练数据
docs/          需求、架构、采集、项目管理和验收文档
models/        模型卡、来源和 SHA；不放权重
schemas/       JSON Schema 与机器可校验合同
scripts/       验证、转换、探针和发布门禁脚本
services/      Pi0.5 双臂服务、YOLO 独立服务
simulation/    V2 场景、Isaac Sim 适配和人工采集入口
src/           总控、Supervisor、执行器、安全和数据合同
tests/         单元、合同、服务和数据管线测试
```

推荐入口：

| 目标 | 文档 |
|---|---|
| 了解 V2 场景和采集顺序 | [`docs/v2-manual-industrial-collection.md`](docs/v2-manual-industrial-collection.md) |
| 了解 Agent 与闭环 | [`docs/architecture/agent-framework.md`](docs/architecture/agent-framework.md) |
| 了解服务接口 | [`docs/architecture/interface-contracts.md`](docs/architecture/interface-contracts.md) |
| 了解仓库放置规则 | [`docs/repository-structure.md`](docs/repository-structure.md) |
| 了解贡献、Issue 和 PR 规则 | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 了解计划、WBS 和验收缺口 | [`docs/project-management/`](docs/project-management/) |

## 8. 协作与发布规则

- 先创建 Issue，再使用短分支和 PR；禁止直接 push `main`；
- 模型权重、训练数据、录像、缓存、密钥和个人机器路径不得进入普通 Git 历史；
- 每个正式数据或模型制品必须有不可变 Manifest 和 SHA-256；
- 练习 Episode 默认不可训练，必须通过预检、终局、回放、GT 隔离和数据 QA；
- 静态 PASS 只能证明静态合同通过，不能写成真实执行或完整训练验收；
- 当前仓库尚未声明开源许可证。对外分发前必须确认比赛规则、上游模型/资产许可
  和团队授权，并通过独立 PR 添加合适的 `LICENSE`。

更详细的协作规范见 [`CONTRIBUTING.md`](CONTRIBUTING.md) 和
[`docs/project-management/dashboard.md`](docs/project-management/dashboard.md)。
