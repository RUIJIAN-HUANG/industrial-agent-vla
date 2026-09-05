# ADR-0005：由同一个 π0.5 服务控制 Arm_A 与 Arm_B

状态：已实现（2026-08-29）

## 决策

Arm_A 和 Arm_B 不再绑定不同 VLA。两个阶段都使用 `pi05` 执行器，由任务子项的
`metadata.arm_id` 选择对应状态流和固定顶视相机：

- `Arm_A` → `robot.arm_a` + `camera.arm_a_rgb` (`CAM_A_TOP`)
- `Arm_B` → `robot.arm_b` + `camera.arm_b_rgb` (`CAM_B_TOP`)

服务部署只保留一个 π0.5 容器和 YOLO 容器。`B_ONLY` 生命周期令牌、交接验证、
动作安全边界和每步重新观测保持不变；模型复用不等于两臂并行。

## 接口变化

- π0.5 infer 请求增加顶层 `arm_id`，并在 `model_input` 中重复携带该字段，避免
  transport 或 backend 丢失控制臂身份。
- `CanonicalRecorder` 和 Canonical V2 action identity 将两只手臂都绑定为 `pi05`。
- 生产 Compose、`.env` 示例和 deployment preflight 只保留 π0.5 与 YOLO 的模型服务。

## 迁移注意

同一个 π0.5 checkpoint 必须用 Arm_A 与 Arm_B 的真实观测/动作数据分别验证。当前
代码只保证协议、路由和安全边界正确，不代表双臂模型训练或真实闭环验收已经完成。
旧模型服务和旧编排入口已从当前仓库删除。
