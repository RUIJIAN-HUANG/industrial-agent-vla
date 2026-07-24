# π0.5 / openpi Service

负责人：E。当前状态：接口占位，真实模型尚未集成。

本目录只放 π0.5/openpi 独立服务的生产代码、依赖、示例配置和测试。实现前必须
固定并记录：

- 上游仓库 Commit、checkpoint SHA-256、norm stats SHA-256；
- LeRobot 数据映射、相机顺序、语言字段和任务 ID；
- JAX 推理输出到统一 `N×7` 动作的转换；
- `/health`、`/v1/infer`、`/v1/cancel` 的超时、错误码和幂等语义；
- 与 `schemas/executor-*.schema.json`、`action-chunk.schema.json` 的契约测试。

不要在此目录提交 checkpoint、训练数据、缓存或个人机器路径。完整接口见
[`../../docs/architecture/interface-contracts.md`](../../docs/architecture/interface-contracts.md)。
