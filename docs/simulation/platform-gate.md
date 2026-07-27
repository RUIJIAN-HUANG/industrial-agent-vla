# G0 - Isaac Sim 5.1 平台验收报告

> Gate：G0
>
> Owner：B；复核：F
>
> 日期：待填写
>
> 状态：`DRAFT - 自动脚本三次均 PASS 且人工检查通过后才能改为 PASS`

## 1. 验收结论

| 验收项 | 门槛 | 实际结果 | 结论 |
|---|---:|---:|---|
| Isaac Sim 独立启动 | 连续 3 次 | 待填写 | 待填写 |
| Headless 物理步 | 1000 步 | 待填写 | 待填写 |
| 场景必要 Prim | 双 Franka、4 零件、1 料箱、3 相机 | 待填写 | 待填写 |
| Reset | 连续 20 次 | 待填写 | 待填写 |
| 数值稳定性 | 无 NaN/Inf、物体未离开工作区 | 待填写 | 待填写 |
| 双臂状态 | 关节名、位置、速度均可读 | 待填写 | 待填写 |
| 相机样本 | 三台相机各 1 帧 | 待填写 | 待填写 |
| 在线 GT 隔离 | 观测与相机清单不包含 GT | 待填写 | 待填写 |
| GUI 人工复核 | 无明显穿模、弹飞、空机器人 | 待填写 | 待填写 |

最终结论：`PASS / FAIL / BLOCKED`

## 2. 冻结配置

| 项目 | 值 |
|---|---|
| Isaac Sim | 5.1.x，完整版本待填写 |
| 场景 | `single_bin_scene_v1` |
| 机器人 | `Arm_A`、`Arm_B`，均为 Franka |
| 相机 | `CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP` |
| 交接 | `A_ONLY -> HANDOFF_VERIFY -> B_ONLY` |
| 场景配置 | `simulation/configs/single_bin_scene_v1.json` |
| 生成场景 | `simulation/generated/single_bin_scene_v1.usda` |
| 证据目录 | `artifacts/g0/<时间>/` |

## 3. 三次独立启动

从 `restart-summary.tsv` 原样复制，不手工修改退出码。

| 次数 | 开始时间 | 退出码 | 用途 | 结论 |
|---:|---|---:|---|---|
| 1 | 待填写 | 待填写 | 1000 步、20 Reset、三相机 | 待填写 |
| 2 | 待填写 | 待填写 | 冷启动 smoke | 待填写 |
| 3 | 待填写 | 待填写 | 冷启动 smoke | 待填写 |

通过条件：三个退出码均为 `0`，三个 `run_result.json` 的 `status` 均为
`PASS`。

## 4. 1000 步与 20 次 Reset

从 `restart-1/run_result.json` 填写：

| 字段 | 实际值 |
|---|---|
| `headless_steps_completed` | 待填写 |
| `headless_elapsed_seconds` | 待填写 |
| `steps_per_second` | 待填写 |
| `resets_completed` | 待填写 |
| `reset_settle_steps` | 待填写 |
| `online_gt_included` | 必须为 `false` |

检查 `restart-1/reset_report.json`：

- [ ] 共 20 条 Reset 记录。
- [ ] 每条 `errors` 都是空列表。
- [ ] Arm_A、Arm_B 关节位置和速度均为有限数。
- [ ] P01-P04 与 Bin_01 未离开工作区。

## 5. 三相机与机器人观测

| 相机 | 样本 | 已人工打开 | 画面结论 |
|---|---|---|---|
| `CAM_A_TOP` | `restart-1/cameras/CAM_A_TOP.ppm` | [ ] | 待填写 |
| `CAM_HANDOFF` | `restart-1/cameras/CAM_HANDOFF.ppm` | [ ] | 待填写 |
| `CAM_B_TOP` | `restart-1/cameras/CAM_B_TOP.ppm` | [ ] | 待填写 |

人工检查：

- [ ] 三张图不是全黑、纯色或重复画面。
- [ ] A 区相机能看到零件和装箱区域。
- [ ] 交接相机能看到 `HANDOFF_CENTER`。
- [ ] B 区相机能看到交接区和 `FINISHED_01`。
- [ ] `robot_observation.json` 只有机器人遥测，没有物体 GT、目标坐标或抓取点。

## 6. GUI 人工复核

自动检查不能代替以下人工观察：

- [ ] `/World/Robots/Arm_A` 和 `/World/Robots/Arm_B` 都显示真实 Franka，
      不是空 Xform。
- [ ] 两台 Franka 基座位置、朝向正确。
- [ ] P01-P04、料箱和桌面无初始穿模。
- [ ] 连续运行至少 30 秒，无零件弹飞和仿真崩溃。
- [ ] 三个 Camera Prim 的视锥方向正确。
- [ ] 保存并重新打开 `single_bin_scene_v1.usda` 后资产仍完整。

GUI 截图/短视频位置：待填写。

## 7. 失败与处置

| 时间 | 失败现象 | 日志 | 原因 | 处理 | 是否复测 |
|---|---|---|---|---|---|
| 待填写；无失败则写“无” | | | | | |

若 Gate 未通过，状态必须保持 `FAIL` 或 `BLOCKED`，并按计划删减渲染质量/
材质，修复唯一 Isaac Sim 路径；不得另开 Gazebo 生产线来规避问题。

## 8. 证据与签字

| 项目 | 值 |
|---|---|
| 证据 SHA 文件 | `artifacts/g0/<时间>/SHA256SUMS.txt` |
| Issue | 待填写 |
| Draft PR | 待填写 |
| B 结论 | 待填写 |
| F 复核 | 待填写 |
| A 的 Gate 决策 | 待填写 |
