# 源代码

当前可运行包为 `industrial_agent`，只包含轻量 Supervisor 核心，不导入真实
VLA、Isaac Sim 或机器人 SDK。冻结主线是三 Agent、双 Franka 固定串行：
π0.5 通过请求中的 `arm_id` 控制 Arm_A 或 Arm_B，YOLO 是同步调用、失败非门控
的评分 sidecar。两个机械臂共用同一个 π0.5 服务，但仍由生命周期令牌保证串行独占。
模块说明、状态机和接入边界见：

- [`../docs/architecture/agent-framework.md`](../docs/architecture/agent-framework.md)
- [`../docs/architecture/interface-contracts.md`](../docs/architecture/interface-contracts.md)
- [`../services/yolo/README.md`](../services/yolo/README.md)

D/E 的真实 VLA 服务、F 的 YOLO 服务和 B 的 Isaac Sim 环境适配应保持独立
进程与独立依赖，依照统一合同接入，不要把 CUDA、PyTorch 与 JAX 依赖装进
核心包。

真实部署使用 `build_executors_from_config(config, transport_factory)` 将配置 URL
绑定到 transport，再交给 `V2Supervisor`。启动时会 fail-closed
校验执行器名称、固定职责、动作合同、checkpoint SHA、norm stats SHA、
令牌顺序和双臂安全字段。在线观测中的退出状态统一为
`robot.arm_a.retreated` 与 `robot.arm_b.retreated`。
