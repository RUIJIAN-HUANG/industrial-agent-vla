# Model Services

> 正式场景边界：V2 `single_bin_manual_industrial_v2`。生产总控只装配 π0.5；
> OpenVLA-OFT 与旧 YOLO/V1 编排服务不属于当前正式运行图。

模型运行时必须与轻量总控 Agent 分离。正式系统由总控、π0.5 和 Isaac 环境边界
组成；在线终局提供器只输出传感证据，不读取 GT。

| 目录 | 责任角色 | 内容 |
|---|---|---|
| `yolo/` | F | 同帧目标检测、原始 bbox/分类/时延归档与独立评分服务合同 |
| `openvla_oft/` | D | OpenVLA-OFT 训练、推理服务和相机适配 |
| `pi05/` | E | π0.5/openpi 训练、推理服务和动作适配 |

正式 V2 调用链：

1. 总控验证 V2 task_id 与冻结指令一一对应；
2. π0.5 接收 Arm_A 图像与状态并返回 7D 动作；
3. 总控安全执行一个动作并重新观测；
4. 在线终局证据达到 3 帧至少 2 票后结束，否则在预算内继续。

不设置 NLP Agent；Supervisor 不解释或改写两个固定指令。两个 VLA 都必须完成
各自固定角色的工业场景微调。一个 VLA 的失败只能在自己的阶段有界重试，不得
由另一个 VLA 接管其机械臂。

同一新鲜图像由当前同步 sidecar 调用 `yolo/` 形成评分证据。YOLO 不得直接调用 VLA、改变
固定角色或发放令牌；其 DetectionPacket 不是 VLA 前置条件，空检测、超时和
坏响应不得阻塞 VLA。
任何物理动作后应对新帧再次检测，以保持原始预测和控制 trace 的同帧关联。

每个服务使用自己的环境、依赖清单、配置和测试，并严格实现
[`../docs/architecture/interface-contracts.md`](../docs/architecture/interface-contracts.md)。
三个路由的框架无关入口核心分别位于
`pi05/handler.py`、`openvla_oft/handler.py` 与 `yolo/handler.py`；它们是模型
backend 前不可绕过的 CAS 像素解析边界。
模型权重、norm stats 实体和训练数据不得提交到 Git；仓库只保存固定 SHA、来源、
下载说明和兼容性信息。

在线 YOLO 容器只能访问图像和任务提示。冻结 GT 只允许由离线
`scripts/evaluate_detection_map.py` 读取，禁止挂载到任何 Agent 服务。
