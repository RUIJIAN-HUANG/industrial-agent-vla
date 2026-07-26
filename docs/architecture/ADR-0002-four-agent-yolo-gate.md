# ADR-0002：四 Agent 同帧 YOLO 感知门（已废止）

- 状态：Superseded
- 被替代：[`ADR-0003-yolo-scoring-sidecar.md`](ADR-0003-yolo-scoring-sidecar.md)
- 日期：2026-07-24
- 决策人：A（项目负责人）
- 影响范围：总控 Agent、YOLO Agent、OpenVLA-OFT Agent、π0.5 Agent
- 不影响：两份官方 PDF 原文及其哈希

> 历史说明：本文保留用于解释方案演进，不再作为实现或验收依据。当前冻结基线
> 明确规定 YOLO 是同步、失败非门控的评分 sidecar；YOLO 空检测、超时、低置信度或坏响应不得
> 阻止 VLA 推理，也不得触发 VLA 切换。

## 背景

官方评价包含物体感知识别精度（mAP 等）和推理速度。VLA 的职责是根据视觉、
语言和机器人状态生成动作，不能代替可重算的目标检测指标。项目因此采用四 Agent
拓扑，并要求每个 VLA 决策周期都绑定一次独立 YOLO 推理。

“同时调用”在控制语义上表示同一个决策周期、同一份新鲜观测和同一个 trace。
默认执行依赖为 YOLO 先返回检测结果，再调用被选中的 VLA；否则当前 VLA 无法使用
bbox/ROI，YOLO 只能成为旁路打分器。

## 决策

固定四个 Agent：

1. 总控 Agent：TaskPlan、FSM、感知门、VLA 路由、安全、恢复和遥测。
2. YOLO Agent：检测、分类、bbox/ROI、可选跟踪及视觉状态属性。
3. OpenVLA-OFT Agent：生成统一 `N×7 ActionChunk`。
4. π0.5 Agent：生成统一 `N×7 ActionChunk`。

环境适配器、PostconditionVerifier 和离线 mAP Evaluator 都是组件或工具，不增加
Agent 数量。

每个动作决策执行以下强制序列：

```text
fresh observation
  -> ObservationGateway
  -> YOLO Agent
  -> validate DetectionPacket
  -> select exactly one VLA
  -> full image + DetectionPacket/ROI + robot state + instruction
  -> validate ActionChunk
  -> Safety
  -> execute one receding-horizon action
  -> acquire a new observation and repeat
```

总控在 VLA 调用前必须验证：

- `DetectionPacket.observation_id` 等于当前观测 ID；
- `DetectionPacket.image_sha256` 等于 VLA 将使用的图像哈希；
- YOLO checkpoint、类别表和配置均使用不可变 SHA-256 标识；
- bbox 数值有限、坐标顺序正确且位于对应图像尺寸内；
- 目标选择结果可追溯到唯一 `detection_id`；
- 在线载荷不含 GT、标注、真实目标位姿或抓取点。

VLA 接收原始全图，并把 bbox/ROI 作为附加感知上下文。只传裁剪图会丢失机械臂、
容器、障碍物和工作空间关系，因此不作为默认模式。

## mAP 证据边界

在线 YOLO 只产生预测。检测框本身不是 mAP；mAP 必须由离线评测器使用冻结测试集
的 GT 与保存的原始预测计算。

```text
online image -> YOLO Agent -> raw predictions.jsonl -> Supervisor/VLA
frozen GT ------------------------------------------> offline mAP evaluator
```

GT 只能进入离线评测进程，不能进入总控、YOLO、任一 VLA 或在线核验器。评测至少
记录 AP50、AP75、AP@[0.50:0.95]、各类别 AP、Precision/Recall，以及固定硬件和
Batch=1 下的推理 P50/P95 延迟。官方未规定唯一 IoU/mAP 口径时，报告必须明确声明
所采用口径，不得把内部阈值表述为官方阈值。

## 失败和恢复

- YOLO 超时、无目标、低置信度、坏响应或陈旧帧使用独立感知重试预算。
- 感知失败不消耗 VLA 同策略重试或 VLA 切换预算。
- 动作尚未执行时，感知预算耗尽后任务失败并保留证据。
- 动作已经执行后，新鲜感知不可用时进入安全停止，禁止沿用旧框或旧 ActionChunk。
- VLA 切换可复用同一未执行观测的已验证 DetectionPacket；任何动作执行后必须
  重新采集图像并重新调用 YOLO。

## 遥测与验收

一次完整决策 trace 至少包含：

- `perception.requested/completed/failed`；
- observation、image、checkpoint、class-map 和 config SHA；
- detection 数量、目标 `detection_id`、bbox 和感知延迟；
- VLA 选择、ActionChunk、安全决定、环境执行和动作后核验；
- 原始预测文件及评测报告的内容哈希。

验收时必须能够从同一个 `trace_id` 追溯“输入帧 -> YOLO 方框 -> VLA 动作 ->
环境结果”，并能够仅凭冻结 GT、原始预测和版本信息重算 mAP。

## 后果

优点：

- 满足检测指标和 VLA 动作能力的职责分离；
- YOLO 能实际帮助 VLA 框定目标，而非只为答辩画框；
- 感知故障不会错误触发 VLA 切换；
- mAP、延迟和端到端动作具备统一证据链。

代价：

- 默认门控模式会增加一次 YOLO 延迟；
- 需要维护类别表、检测数据集、坐标约定和额外服务；
- YOLO 闭集检测仍需通过 ID/OOD 数据划分证明工业场景泛化。
