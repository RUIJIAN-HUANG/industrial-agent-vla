# XH-202607 官方测试与提交验收指南

本指南用于评审前复现、官方测试和最终提交包检查。官方要求以
[`docs/official/`](official/) 中的两份 PDF 为最高依据；本文件只提供仓库内的
可执行流程，不替代官方原文。

## 1. 测试范围

正式测试覆盖三项冻结任务：

| 任务 | `task_id` | 指令 |
|---|---|---|
| 任务一：零件装箱 | `P01_TO_S11` | `把P01放到S11中` |
| 任务二：工具装箱 | `W01_TO_S14` | `把W01放到S14中` |
| 任务三：料箱交接搬运 | `BIN01_TO_FINISHED01` | `把Bin_01搬到FINISHED_01` |

系统具备 π0.5 与 YOLO 双模型推理能力。最终交付时，π0.5 checkpoint、norm
stats 和 YOLO 权重不进入普通 Git 历史，而是放入同一个提交包，并由 SHA-256
和运行配置绑定。

每个正式回合必须满足：

- 使用冻结的 `task_id`、原始指令、对象和槽位，不自行改写指令；
- 每个 7D 微动作后获取新鲜 Observation；
- 终局使用三个去重帧，至少两票通过；
- 观测错误、动作越界、服务故障或超时必须安全停止；
- 任务三必须留存 `A_ONLY → HANDOFF_VERIFY → B_ONLY` 的交接事件和双臂互斥记录；
- 在线推理链路不得读取 GT、目标坐标、轨迹点或抓取姿态。

## 2. 环境准备

代码检查可在 Python 3.10+ 环境执行；Isaac Sim 验收需要 Linux Isaac Sim 5.1
目标机。模型服务目标机还需要 NVIDIA 驱动、Docker Compose v2 和 NVIDIA
Container Toolkit。

准备以下外部制品，两个模型权重最终一起放入提交包：

```text
<PI05_CHECKPOINT_DIR>/       π0.5 checkpoint 目录
<PI05_NORM_STATS>.json       π0.5 norm stats
<YOLO_CHECKPOINT>.pt         YOLO 权重
<PI05_IMAGE_TAR>             π0.5 服务镜像 tar
<YOLO_IMAGE_TAR>             YOLO 服务镜像 tar
```

先验证官方基线和仓库卫生：

```powershell
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python scripts/check_repository_hygiene.py
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

`scripts/run_mock_demo.py` 只能验证调用顺序、令牌和错误处理，不能作为真实
π0.5、YOLO 或 Isaac Sim 验收证据。

## 3. 模型服务与制品预检

使用真实制品构建自包含提交包。输出目录必须位于仓库外，且此前不存在：

```powershell
python scripts/build_submission_bundle.py build `
  --output-dir D:\submission\XH-202607-final `
  --pi05-checkpoint-dir <PI05_CHECKPOINT_DIR> `
  --pi05-norm-stats <PI05_NORM_STATS> `
  --yolo-checkpoint <YOLO_CHECKPOINT> `
  --pi05-image-tar <PI05_IMAGE_TAR> `
  --yolo-image-tar <YOLO_IMAGE_TAR> `
  --pi05-image industrial-agent/pi05:submission `
  --pi05-image-digest sha256:<PI05_IMAGE_DIGEST> `
  --yolo-image industrial-agent/yolo:submission `
  --yolo-image-digest sha256:<YOLO_IMAGE_DIGEST> `
  --pi05-gpu-ids 0 `
  --yolo-gpu-id 1
```

构建完成后，在移动或解压后的提交包中执行：

```powershell
python scripts/build_submission_bundle.py verify --bundle-dir <BUNDLE_DIR>
python scripts/build_submission_bundle.py prepare-env --bundle-dir <BUNDLE_DIR>
.\verify.ps1
```

在模型机启动服务前，先运行资产预检：

```bash
python deploy/preflight.py \
  --env-file deploy/.env.production \
  --phase assets \
  --output artifacts/deployment/model-assets-preflight.json
```

预检必须核对 π0.5 checkpoint、norm stats、YOLO 权重、类别表、镜像 digest、
GPU、端口和共享 CAS。任何失败都不得启动 real 模式服务。

服务启动后再次检查身份：

```bash
python deploy/preflight.py \
  --env-file deploy/.env.production \
  --phase services \
  --output artifacts/deployment/model-services-preflight.json
```

两个服务的 `/health` 必须返回 `status=ready`、非 Mock 模式以及与提交包清单
一致的 checkpoint、norm-stats、class-map 和配置 SHA。

## 4. Isaac Sim 场景验收

所有证据写入独立目录，例如 `artifacts/official-test/`；每次运行使用新的子目录。

### 4.1 静态场景合同

```powershell
python simulation/run_v2_scene_acceptance.py `
  --evidence-dir artifacts/official-test/static
```

### 4.2 GUI、HOME、IK 和双臂微动作

以下命令在 Isaac Sim 5.1 目标机执行：

```powershell
python simulation/run_v2_gui_scene_acceptance.py `
  --output-scene artifacts/official-test/gui/v2-scene.usda `
  --evidence-dir artifacts/official-test/gui

python simulation/run_v2_home_acceptance.py `
  --output-scene artifacts/official-test/home/v2-home.usda `
  --evidence-dir artifacts/official-test/home

python simulation/run_v2_ik_reachability_acceptance.py `
  --output-scene artifacts/official-test/ik/v2-ik.usda `
  --evidence-dir artifacts/official-test/ik

python simulation/run_v2_dual_arm_micro_motion_acceptance.py `
  --output-scene artifacts/official-test/micro-motion/v2-micro-motion.usda `
  --evidence-dir artifacts/official-test/micro-motion
```

### 4.3 Headless 稳定性

```powershell
python simulation/run_v2_stability_acceptance.py `
  --output-scene artifacts/official-test/stability/v2-stability.usda `
  --evidence-dir artifacts/official-test/stability `
  --steps 1000 `
  --resets 20
```

报告必须记录启动、reset、物理步、相机样本、NaN/Inf、碰撞/越界和退出码。

## 5. 三项正式任务测试

每项任务至少保存以下材料：

```text
task.json                 冻结 task_id、原始指令和配置摘要
run-result.json           成功/失败、终局票数、停止原因
events.jsonl              Observation、动作、交接和恢复事件
model-health.json         π0.5/YOLO /health 身份
video.mp4                 从输入到终局的一镜到底录像（如官方要求）
sha256sums.txt            代码、配置、模型和证据摘要
```

π0.5 Isaac 闭环入口示例：

```powershell
python simulation/run_pi05_isaac_closed_loop.py `
  --scene-config simulation/configs/single_bin_scene_v2.json `
  --agent-config configs/agent.default.json `
  --task configs/task.v2.p01-to-s11.example.json `
  --artifact-root artifacts/official-test/task-1 `
  --result-file artifacts/official-test/task-1/run-result.json `
  --headless
```

对任务二替换为 `task.v2.w01-to-s14.example.json`，对任务三替换为
`task.v2.bin01-to-finished01.example.json`。若目标环境使用不同任务配置，必须
在证据中记录配置文件的 SHA-256。

测试人员应分别检查：

1. 任务一：P01 进入 S11，终局核验通过后 Arm_A 安全回 HOME；
2. 任务二：W01 进入 S14，工具类别/槽位/姿态证据一致；
3. 任务三：Arm_A 完成装箱并交接，Arm_B 在 `B_ONLY` 阶段搬运至
   `FINISHED_01`，两臂不得同时进入共享区；
4. 注入抓空、错格、倾倒/倒放、掉落或服务超时后，系统必须记录失败原因、重新
   获取观测并生成有界恢复动作；超过预算或遇到安全异常时必须 safe-stop。

## 6. 官方证据包检查

提交前由第二名测试人员复核：

- 两份官方 PDF SHA-256 与基线一致；
- 三项任务、仿真门禁、模型身份和失败恢复均有可定位证据；
- 证据中的 commit、配置、模型摘要、环境和命令可复现；
- 视频、日志和 JSON 不含 token、个人绝对路径或在线 GT；
- `reports/evidence-index.md` 的状态与实际证据一致；
- 最终提交包内同时存在 π0.5 checkpoint、norm stats、YOLO 权重、两个服务镜像、
  启动脚本、验证脚本和 SHA-256 清单；
- 解压到另一台干净目标机后，先执行 `verify.ps1`，再执行资产预检、服务预检和
  Isaac Sim 无动作验收。

任何一项只有接口测试、Mock 或静态检查而没有真实运行证据时，应标记为
`No evidence` 或 `Partial`，不得标记为 `Reproducible`。

## 7. 结果登记模板

| 项目 | 结果 | 证据路径 | SHA-256 | 复核人 |
|---|---|---|---|---|
| 官方基线 | PASS / FAIL | `artifacts/baseline/` |  |  |
| 仓库卫生与 CI | PASS / FAIL | `artifacts/ci/` |  |  |
| 模型资产预检 | PASS / FAIL | `artifacts/deployment/` |  |  |
| Isaac Sim 门禁 | PASS / FAIL | `artifacts/official-test/` |  |  |
| 任务一 | PASS / FAIL | `artifacts/official-test/task-1/` |  |  |
| 任务二 | PASS / FAIL | `artifacts/official-test/task-2/` |  |  |
| 任务三 | PASS / FAIL | `artifacts/official-test/task-3/` |  |  |
| 提交包复核 | PASS / FAIL | `<BUNDLE_DIR>/evidence/` |  |  |
