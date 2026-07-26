# OpenVLA-OFT Service

负责人：D。当前状态：接口占位，真实模型尚未集成。

冻结定位：OpenVLA-OFT 是 Arm_B 的唯一 VLA。它保持等待，直到 Supervisor
完成三帧交接核验、发布 `handoff_ready` 并授予 `B_ONLY` 令牌；随后接收预设的
下游协作指令、`CAM_B_TOP`/腕部完整图像和 Arm_B 状态，把
`HANDOFF_CENTER` 的满箱搬到 `FINISHED_01`，再退回 `HOME_B`。
YOLO DetectionPacket 不是推理前置条件。OpenVLA-OFT 必须针对这一固定角色完成
工业场景微调并提供 base/tuned 同协议对照，不能仅交付未微调 checkpoint。

本目录只放 OpenVLA-OFT 独立服务的生产代码、依赖、示例配置和测试。实现前必须
固定并记录：

- 上游仓库 Commit、checkpoint SHA-256、norm stats SHA-256；
- 相机顺序、图像尺寸、语言字段和任务 ID；
- 统一 `N×7` 动作到模型原生动作的转换；
- `/health`、`/v1/infer`、`/v1/cancel` 的超时、错误码和幂等语义；
- 与 `schemas/executor-*.schema.json`、`action-chunk.schema.json` 的契约测试。
- 工业微调的数据/配置/checkpoint SHA、base/tuned 成功率与失败分布。
- 服务只能输出 `arm_id=Arm_B` 的动作；未持有 `B_ONLY` 时不得执行。
- 恢复时必须使用 Arm_B 的新鲜观测重新推理，禁止接管 Arm_A 的装箱阶段。

不要在此目录提交 checkpoint、训练数据、缓存或个人机器路径。完整接口见
[`../../docs/architecture/interface-contracts.md`](../../docs/architecture/interface-contracts.md)。
