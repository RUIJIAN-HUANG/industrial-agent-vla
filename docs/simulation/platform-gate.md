# G0 - Isaac Sim 5.1 平台验收报告

> Gate：G0
>
> Owner：B；复核：F
>
> 日期：2026-07-27
>
> 状态：`B PASS - 自动验收与 GUI 人工复核均通过，待 F/A 签署`

## 1. 验收结论

| 验收项 | 门槛 | 实际结果 | 结论 |
|---|---:|---:|---|
| Isaac Sim 独立启动 | 连续 3 次 | 3/3 次启动成功，退出码均为 `0` | PASS |
| Headless 物理步 | 1000 步 | 1000/1000 | PASS |
| 场景必要 Prim | 双 Franka、4 零件、1 料箱、3 相机 | 10/10 个必要 Prim | PASS |
| Reset | 连续 20 次 | 20/20 | PASS |
| 数值稳定性 | 无 NaN/Inf、物体未离开工作区 | 自动检查无错误 | PASS |
| 双臂状态 | 关节名、位置、速度均可读 | `robot_observation.json` 已生成 | PASS |
| 相机样本 | 三台相机各 1 帧 | 3/3 个 PPM 样本已生成 | PASS（自动） |
| 在线 GT 隔离 | 观测与相机清单不包含 GT | `online_gt_included=false` | PASS |
| GUI 人工复核 | 无明显穿模、弹飞、空机器人 | 两台完整 Franka 与场景正常；播放后稳定 | PASS |

最终结论：`PASS（成员 B）`；F 复核与 A 的最终 Gate 决策待签署。

## 2. 冻结配置

| 项目 | 值 |
|---|---|
| Isaac Sim | 5.1.0 |
| 场景 | `single_bin_scene_v1` |
| 机器人 | `Arm_A`、`Arm_B`，均为 Franka |
| 相机 | `CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP` |
| 交接 | `A_ONLY -> HANDOFF_VERIFY -> B_ONLY` |
| 场景配置 | `simulation/configs/single_bin_scene_v1.json` |
| 生成场景 | `simulation/generated/single_bin_scene_v1.usda` |
| 证据目录 | `artifacts/g0/20260727-210649/` |

## 3. 三次独立启动

从 `restart-summary.tsv` 原样复制，不手工修改退出码。

| 次数 | 开始时间 | 退出码 | 用途 | 结论 |
|---:|---|---:|---|---|
| 1 | 2026-07-27T21:06:49+08:00 | 0 | 1000 步、20 Reset、三相机 | PASS |
| 2 | 2026-07-27T21:07:11+08:00 | 0 | 冷启动 smoke | PASS |
| 3 | 2026-07-27T21:07:22+08:00 | 0 | 冷启动 smoke | PASS |

通过条件：三个退出码均为 `0`，三个 `run_result.json` 的 `status` 均为
`PASS`。

## 4. 1000 步与 20 次 Reset

从 `restart-1/run_result.json` 填写：

| 字段 | 实际值 |
|---|---|
| `headless_steps_completed` | `1000` |
| `headless_elapsed_seconds` | `0.40737210999941453` |
| `steps_per_second` | `2454.758132561989` |
| `resets_completed` | `20` |
| `reset_settle_steps` | `120` |
| `online_gt_included` | `false` |

检查 `restart-1/reset_report.json`：

- [x] 共 20 条 Reset 记录。
- [x] 每条 `errors` 都是空列表。
- [x] Arm_A、Arm_B 关节位置和速度均为有限数。
- [x] P01-P04 与 Bin_01 未离开工作区。

## 5. 三相机与机器人观测

| 相机 | 样本 | 已人工打开 | 画面结论 |
|---|---|---|---|
| `CAM_A_TOP` | `restart-1/cameras/CAM_A_TOP.ppm` | [x] | 可见 A 区 4 个红色零件、料箱和 Arm_A；机械臂遮挡较大 |
| `CAM_HANDOFF` | `restart-1/cameras/CAM_HANDOFF.ppm` | [x] | 可见绿色 `HANDOFF_CENTER` 与料箱边缘 |
| `CAM_B_TOP` | `restart-1/cameras/CAM_B_TOP.ppm` | [x] | 可见交接区、黄色 `FINISHED_01` 和 Arm_B |

人工检查：

- [x] 三张图不是全黑、纯色或重复画面。
- [x] A 区相机能看到零件和装箱区域。
- [x] 交接相机能看到 `HANDOFF_CENTER`。
- [x] B 区相机能看到交接区和 `FINISHED_01`。
- [x] `robot_observation.json` 只有机器人遥测，没有物体 GT、目标坐标或抓取点。

画面风险：三台相机曝光偏高，`CAM_A_TOP` 与 `CAM_B_TOP` 存在明显机械臂
近景遮挡。当前画面足以通过 G0 场景覆盖检查，但进入感知识别数据采集前应调整
曝光、灯光和相机位姿。

## 6. GUI 人工复核

自动检查不能代替以下人工观察：

- [x] `/World/Robots/Arm_A` 和 `/World/Robots/Arm_B` 都显示真实 Franka，
      不是空 Xform。
- [x] 两台 Franka 基座位置、朝向正确。
- [x] P01-P04、料箱和桌面无初始穿模。
- [x] 连续运行至少 30 秒，无零件弹飞和仿真崩溃。
- [x] 三个 Camera Prim 的视锥方向正确。
- [x] 自动脚本保存、GUI 重新打开 `single_bin_scene_v1.usda` 后资产仍完整。

GUI 截图：Draft PR 评论附件 `g0-gui-overview.png` 与
`g0-gui-stage-tree.png`。播放开始时两臂进入物理初始化后的稳定关节姿态，
随后保持静止；场景未绑定控制任务，该表现符合 G0 静态场景预期。

## 7. 失败与处置

| 时间 | 失败现象 | 日志 | 原因 | 处理 | 是否复测 |
|---|---|---|---|---|---|
| 2026-07-27 | Isaac Sim 5.1 API 兼容错误 | 历次 G0 `run_result.json` | Stage、USD customData 与 articulation API 在 5.1 中有变化 | 增加 5.1 兼容层与回归测试 | 是，最终 3/3 PASS |

若 Gate 未通过，状态必须保持 `FAIL` 或 `BLOCKED`，并按计划删减渲染质量/
材质，修复唯一 Isaac Sim 路径；不得另开 Gazebo 生产线来规避问题。

## 8. 证据与签字

| 项目 | 值 |
|---|---|
| 证据 SHA 文件 | `artifacts/g0/20260727-210649/SHA256SUMS.txt` |
| 证据压缩包 | `member-b-g0-20260727-210649.tar.gz`，SHA256 `0eb8806c062e58edb44655f2892ef11760de9eb862ef25048abb3487bb1240c1` |
| Issue | 未创建 |
| Draft PR | [#7](https://github.com/RUIJIAN-HUANG/industrial-agent-vla/pull/7) |
| B 结论 | PASS；自动验收与 GUI 人工复核均通过 |
| F 复核 | 待填写 |
| A 的 Gate 决策 | 待填写 |
