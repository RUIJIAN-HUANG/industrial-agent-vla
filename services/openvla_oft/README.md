# OpenVLA-OFT 独立服务

本目录是 `OpenVLA-OFT` 在本项目中的独立服务实现，冻结职责是：

- 只作为 `Arm_B` 的 VLA 执行器；
- 只处理 `S02_ARM_B_TRANSPORT` 阶段；
- 只接收 Supervisor 在三帧交接核验通过并授予 `B_ONLY` 后发来的请求；
- 接收冻结下游协作指令、`CAM_B_TOP`、`CAM_B_WRIST` 和 Arm_B 状态；
- 输出 canonical `N x 7` 动作块，将 `HANDOFF_CENTER` 的满箱搬到 `FINISHED_01`，再退回 `HOME_B`；
- 不接收 YOLO DetectionPacket 作为推理前置条件；
- 不接收 GT、仿真真值、目标坐标、轨迹点或抓取姿态。

当前代码已经完成服务边界、Mock 推理、契约校验、示例配置和测试。真实
OpenVLA-OFT 权重加载、工业场景微调、base/tuned 对照实验仍未集成，不能把
当前 Mock 模式当作比赛最终模型交付。

## 当前完成范围

- 标准库 HTTP 服务入口：`src/openvla_oft/app.py`
- 服务路由和状态管理：`src/openvla_oft/routes.py`
- `/health`、`/v1/infer`、`/v1/cancel` 请求与响应构造：
  `src/openvla_oft/schemas.py`
- CAS 图像引用校验：`src/openvla_oft/cas.py`
- `N x 7` 动作校验与转换：`src/openvla_oft/model.py`、
  `src/openvla_oft/utils.py`
- 公开可复现配置：`configs/agent.default.json`、
  `configs/openvla.default.json`
- 契约测试：`tests/test_health.py`、`tests/test_infer.py`、
  `tests/test_cancel.py`

核心运行时依赖保持为空；测试依赖在 `pyproject.toml` 的 `[project.optional-dependencies].test`
中声明。

## 固定契约

### `GET /health`

返回 `schemas/executor-health.schema.json` 所需字段，包括：

- `service=openvla_oft`
- `status=ready`
- `checkpoint_sha`
- `norm_stats_sha`
- `supported_task_types`
- `supported_action_contracts`

Supervisor 启动 episode 前应核对服务名、动作契约版本、checkpoint SHA 和 norm stats SHA。

### `POST /v1/infer`

请求必须符合 `schemas/executor-infer.schema.json` 的 request envelope。
OpenVLA-OFT 服务额外固定以下约束：

- `executor` 必须是 `openvla_oft`
- `subtask_id` 必须是 `S02_ARM_B_TRANSPORT`
- `model_input.task_description` 必须等于冻结 Arm_B 指令
- `model_input.full_image.camera_id` 必须是 `CAM_B_TOP`
- `model_input.wrist_image.camera_id` 必须是 `CAM_B_WRIST`
- 图像引用必须是 `cas://sha256/<64 hex>`，并与 `image_sha256` 一致
- `state` 至少包含 7 个有限数值
- `checkpoint_sha` 和 `norm_stats_sha` 必须与部署配置一致

成功响应返回 `status=ok` 和 `action_chunk`。失败响应返回 `status=error`
和稳定错误对象。

### `POST /v1/cancel`

请求必须符合 `schemas/executor-cancel.schema.json` 的 request envelope。
取消操作是幂等的：

- 活跃请求存在时返回 `status=cancelled`
- 请求已完成时返回 `status=already_completed`
- 无对应服务上下文时返回 `status=not_found`

## 动作格式

OpenVLA-OFT 服务输出 canonical `N x 7` 动作：

```text
dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad, gripper_norm
```

单位和坐标系固定为：

- `action_space=ee_delta_pose_gripper`
- `frame=robot_base`
- `translation_unit=m`
- `rotation_unit=rad`
- `gripper_unit=normalized`

当前模型原生动作顺序固定为同一 `N x 7` 布局。转换逻辑显式保存在
`openvla_oft.model.ActionConverter`，后续接入真实模型时不得悄悄改变轴顺序、
单位或夹爪范围。

## 配置与制品规则

Git 中只保存公开、可复现配置，不保存 checkpoint、训练数据、缓存、视频、
机器人录包、原始日志或个人机器路径。

需要替换为真实值的字段在 `configs/openvla.default.json` 中记录：

- `upstream.repo`
- `upstream.commit_sha`
- `artifacts.checkpoint_sha`
- `artifacts.norm_stats_sha`
- `fine_tuning.dataset_manifest_sha`
- `fine_tuning.config_sha`
- `fine_tuning.tuned_checkpoint_sha`
- `fine_tuning.base_success_rate`
- `fine_tuning.tuned_success_rate`
- `fine_tuning.failure_distribution`

正式实验应放在仓库 `experiments/` 下，目录名建议：

```text
YYYYMMDD_openvla-oft_s02-arm-b_<short-id>/
```

实验记录应包含 Git commit、配置文件、随机种子、数据清单 SHA、权重 SHA、
运行环境和结果索引；完整权重和数据放外部制品存储。

## 运行 Mock 服务

从仓库根目录执行：

```powershell
cd services/openvla_oft
python -m pip install -e ".[test]"
$env:OPENVLA_OFT_USE_MOCK = "1"
python scripts/run_service.py --host 127.0.0.1 --port 8102
```

Mock 模式只用于接口和联调 smoke，不代表真实 OpenVLA-OFT 微调模型已完成。

## 真实模型接入入口

真实推理应接入 `src/openvla_oft/model.py` 中的 `RealOpenVLAPolicy`。
接入前必须先固定并记录：

- 上游 OpenVLA-OFT 仓库 commit；
- tuned checkpoint SHA-256；
- norm stats SHA-256；
- 工业微调数据 manifest SHA；
- 微调配置 SHA；
- base/tuned 同协议成功率与失败分布。

真实模型路径不得写入配置文件，应通过环境变量或制品管理系统提供：

```powershell
$env:OPENVLA_OFT_USE_MOCK = "0"
$env:OPENVLA_OFT_CHECKPOINT_DIR = "<external-artifact-store>/openvla_oft/checkpoint"
$env:OPENVLA_OFT_CHECKPOINT_SHA = "sha256:<64 hex>"
$env:OPENVLA_OFT_NORM_STATS_SHA = "sha256:<64 hex>"
$env:CAS_ROOT = "<external-cas-root>"
```

## 测试

服务自身测试：

```powershell
python -m pytest services/openvla_oft/tests -q
```

项目级检查：

```powershell
python scripts/check_repository_hygiene.py
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
git diff --check
```

本次验证结果：

- `python -m pytest -q`：`108 passed`
- `python -m unittest discover -s tests -v`：`108 tests OK`
- `ruff format --check .`：通过
- `ruff check .`：通过
- 官方 PDF 基线校验：通过
- 项目冻结输入校验：通过
- 仓库卫生检查：通过
- `scripts/run_mock_demo.py`：三个场景均 `success=true`

## 当前未完成项

- 未接入真实 OpenVLA-OFT 权重；
- 未完成工业场景微调；
- 未提供 base/tuned 对照实验结果；
- 未提交真实 checkpoint/norm stats/data/config SHA；
- 未完成端到端 Isaac Sim 真实控制闭环。

这些缺口必须在后续实验和集成 PR 中补齐，不能用 Mock checkpoint 或占位 SHA
替代正式验收证据。
