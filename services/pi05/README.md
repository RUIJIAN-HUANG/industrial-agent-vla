# π0.5 / openpi Service

负责人：E。当前状态：接口占位，真实模型尚未集成。

冻结定位：π0.5 是 Arm_A 的唯一 VLA。它直接接收预设的上游自然语言、
`CAM_A_TOP`/腕部完整图像和 Arm_A 状态，负责抓取四个零件、纠正倒放零件、
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

不要在此目录提交 checkpoint、训练数据、缓存或个人机器路径。完整接口见
[`../../docs/architecture/interface-contracts.md`](../../docs/architecture/interface-contracts.md)。
