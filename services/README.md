# Model Services

真实 VLA 运行时必须与轻量总 Agent 分离，避免 PyTorch/CUDA 与 JAX/openpi 的依赖
冲突：

| 目录 | 责任角色 | 内容 |
|---|---|---|
| `openvla_oft/` | D | OpenVLA-OFT 训练、推理服务和相机适配 |
| `pi05/` | E | π0.5/openpi 训练、推理服务和动作适配 |

每个服务使用自己的环境、依赖清单、配置和测试，并严格实现
[`../docs/architecture/interface-contracts.md`](../docs/architecture/interface-contracts.md)。
模型权重、norm stats 实体和训练数据不得提交到 Git；仓库只保存固定 SHA、来源、
下载说明和兼容性信息。
