# Model Services

模型运行时必须与轻量总控 Agent 分离，避免 YOLO/PyTorch/CUDA 与 JAX/openpi
的依赖冲突。系统固定为四个 Agent：总控、YOLO、OpenVLA-OFT 和 π0.5；
Verifier、环境和离线 mAP 评测器不是 Agent。

| 目录 | 责任角色 | 内容 |
|---|---|---|
| `yolo/` | F | 同帧目标检测、原始 bbox/分类/时延归档与独立评分服务合同 |
| `openvla_oft/` | D | OpenVLA-OFT 训练、推理服务和相机适配 |
| `pi05/` | E | π0.5/openpi 训练、推理服务和动作适配 |

两个 VLA 在同一任务中固定串行调用，不做在线路由：

1. 总控把预设的上游自然语言、Arm_A 图像和状态原样交给 `pi05/`；
2. π0.5 完成四零件装箱、把料箱放到 `HANDOFF_CENTER` 并让 Arm_A 退出；
3. Supervisor 用三张新鲜图像完成至少两票交接核验，再发布
   `handoff_ready` 并把令牌切为 `B_ONLY`；
4. 总控把预设的下游协作指令、Arm_B 图像和状态原样交给
   `openvla_oft/`，由其把满箱搬到 `FINISHED_01`。

不设置 NLP Agent；Supervisor 不解释或改写两个固定指令。两个 VLA 都必须完成
各自固定角色的工业场景微调。一个 VLA 的失败只能在自己的阶段有界重试，不得
由另一个 VLA 接管其机械臂。

同一新鲜图像由当前同步 sidecar 调用 `yolo/` 形成评分证据。YOLO 不得直接调用 VLA、改变
固定角色或发放令牌；其 DetectionPacket 不是 VLA 前置条件，空检测、超时和
坏响应不得阻塞 VLA。
任何物理动作后应对新帧再次检测，以保持原始预测和控制 trace 的同帧关联。

每个服务使用自己的环境、依赖清单、配置和测试，并严格实现
[`../docs/architecture/interface-contracts.md`](../docs/architecture/interface-contracts.md)。
模型权重、norm stats 实体和训练数据不得提交到 Git；仓库只保存固定 SHA、来源、
下载说明和兼容性信息。

在线 YOLO 容器只能访问图像和任务提示。冻结 GT 只允许由离线
`scripts/evaluate_detection_map.py` 读取，禁止挂载到任何 Agent 服务。
