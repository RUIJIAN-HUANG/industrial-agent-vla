# Source Code

当前可运行包为 `industrial_agent`，只包含轻量总 Agent 核心，不导入真实 VLA、
Isaac/Gazebo 或机器人 SDK。模块说明、状态机和接入边界见：

- [`../docs/architecture/agent-framework.md`](../docs/architecture/agent-framework.md)
- [`../docs/architecture/interface-contracts.md`](../docs/architecture/interface-contracts.md)

D/E 的真实模型服务和 B 的环境适配应保持独立进程/独立依赖，依照统一合同
接入，不要把 CUDA、PyTorch 与 JAX 依赖装进核心包。
