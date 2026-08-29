# 当前冻结场景与流程

当前场景是 `single_bin_manual_industrial_v2`，包含两台 Franka、三路 RGB 相机、
一个料箱和八个工业零件。两台机械臂由同一个 π0.5 服务按令牌串行控制。

## 流程

1. 总控校验冻结任务和在线观测。
2. YOLO 可选地对同一帧生成检测证据；检测不是 π0.5 推理前置条件。
3. 总控向 π0.5 发送 `arm_id=Arm_A` 或 `arm_id=Arm_B`。
4. 安全边界通过后执行一个 7D 动作，再重新观测。
5. 后置条件通过固定票数后，总控结束任务或发放下一个控制令牌。

交接证据事件顺序固定为 `handoff.verified` → `handoff.ready`；候选检查不直接
授予 Arm_B 控制权。

`Arm_B` 是否运动由任务合同和控制令牌决定，不由服务名称决定。正式的 P01/W01
单件任务当前使用 Arm_A；料箱搬运阶段使用 Arm_B，并仍调用同一个 π0.5。

## 场景真源

- `simulation/configs/single_bin_scene_v2.json`
- `configs/v2-task-profile.json`
- `configs/agent.v2.default.json`
- `schemas/online-observation-v2.schema.json`
