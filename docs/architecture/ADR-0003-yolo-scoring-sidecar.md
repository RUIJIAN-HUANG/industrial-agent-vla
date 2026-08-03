# ADR-0003：四 Agent 双臂固定串行与 YOLO 同步评分 sidecar

- 状态：Accepted
- 日期：2026-07-26
- 决策人：A（项目负责人）
- 替代：[`ADR-0002-four-agent-yolo-gate.md`](ADR-0002-four-agent-yolo-gate.md)
- 影响范围：Supervisor、π0.5 Agent、OpenVLA-OFT Agent、YOLO Agent、双臂安全交接与离线评测
- 不影响：两份官方 PDF 原文及其哈希

## 背景

比赛既要求完整的多智能体闭环，也要求可重算的目标检测 mAP 和推理速度证据。
VLA 自带的视觉编码能力不能替代独立目标检测评测，因此必须保留 YOLO Agent。
同时，YOLO 漏检、超时或坏响应不能阻止 VLA 根据完整图像继续完成固定任务。

项目当前只采用一个可交付主线：四个 Agent、两台 Franka、单箱、固定交接位和
固定串行协作。不存在按任务复杂度选择执行器，也不存在一个 VLA 失败后由另一个
VLA 接管其机械臂或子任务。

## 决策

### 1. 四 Agent 的固定职责

| Agent | 固定职责 | 明确禁止 |
|---|---|---|
| Supervisor | 读取冻结 `TaskProfile`，管理 FSM、令牌、安全、超时、有限重试、事件与证据关联 | 解析自然语言、生成抓取坐标、改变 VLA 固定职责 |
| π0.5 Agent | 接收预设自然语言和 Arm_A 在线观测，控制 Arm_A 装箱、移箱到 `HANDOFF_CENTER` 并退出共享区 | 控制 Arm_B、承担成品搬运 |
| OpenVLA-OFT Agent | 在 `handoff.ready` 已持久化且令牌为 `B_ONLY` 后，控制 Arm_B 把同一料箱搬到 `FINISHED_01` 并退出 | 控制 Arm_A、提前进入共享区 |
| YOLO Agent | 对当前新鲜 RGB 帧输出检测框、类别、置信度、时延和可离线重算 mAP 的原始预测 | 生成 VLA 动作、读取在线 GT、授予控制令牌 |

### 2. 双臂固定串行生命周期

```text
RESET
  → A_ONLY
  → π0.5 / Arm_A 装箱并把 Bin_01 放到 HANDOFF_CENTER
  → robot.arm_a.retreated = true
  → handoff.candidate_checked（A_ONLY 下预检，不计票）
  → HANDOFF_VERIFY（锁定双臂）
  → 锁臂后恰好三张不同的新鲜观测达到 2/3 复合票数
  → 持久化 handoff.verified
  → 持久化 handoff.ready
  → B_ONLY
  → OpenVLA-OFT / Arm_B 把 Bin_01 搬到 FINISHED_01
  → robot.arm_b.retreated = true
  → NONE / SUCCEEDED
```

任一时刻共享区最多允许一台机械臂进入。`HANDOFF_VERIFY` 阶段不允许任一 VLA
产生动作。当前固定子任务失败时，只能在该子任务的恢复预算内重新观察并重试；
`handoff.candidate_checked` 和 `handoff.verified` 都不是就绪事件，只有 durable
`handoff.ready` 能授权 Arm_B。冻结自然语言中的 `handoff_ready` 仅是业务信号名，
不得用作事件类型。
预算耗尽后安全停止，不把子任务交给另一个 VLA。

### 3. YOLO 是同步调用、失败非门控的评分 sidecar

当前实现会在控制循环中对同一新鲜观测同步调用一次 YOLO，以便稳定关联
`trace_id + observation_id + image_sha256`。这是当前实现顺序，不宣称 YOLO
与 VLA 真正异步或并行执行。

同步调用不等于控制硬门：

- YOLO 成功时，保存全部候选 bbox、类别、置信度、分阶段/总时延和部署 SHA；
- 空检测是合法成功预测，必须原样保存；
- YOLO 超时、不可用、低置信度、队列拥塞或坏响应只记录 sidecar 失败；
- YOLO 成功不是调用 VLA、推进固定阶段或发放令牌的前置条件；
- `A_ONLY → HANDOFF_VERIFY → B_ONLY` 只由 Supervisor 根据冻结 FSM、在线观测、
  机器人遥测和安全条件管理；
- YOLO 故障不得消耗 VLA 重试预算，也不得改变 π0.5 与 OpenVLA-OFT 的固定职责；
- 默认不把 bbox 注入 VLA。未来若验证 bbox 提示有收益，必须通过新 ADR 审批，
  且仍不得成为控制硬门。

因此，当前 sidecar 会带来一个有界的同步调用时延，但不会因为检测失败而拒绝
VLA 动作或交接令牌。

### 4. mAP 证据边界

在线 YOLO 只产生预测，不能在线给出可信 mAP。原始预测与冻结 GT 只允许在隔离
的离线评测进程汇合：

```text
在线 RGB → YOLO → yolo-evidence.jsonl → raw_predictions.json ─┐
                                                               ├→ 离线评测 → detection_metrics.json
冻结数据集 → COCO GT annotations.json ─────────────────────────┘
```

其中 `yolo-evidence.jsonl` 是在线检测包、空检测和失败记录的 durable 原始证据；
`raw_predictions.json` 是离线评测器保存的精确预测副本；
`detection_metrics.json` 包含 AP50、AP75、mAP50:95、Precision/Recall 和时延指标。

GT、人工标注框、oracle 状态、真实目标位姿和抓取点禁止进入 Supervisor、
YOLO、任一 VLA、在线 Verifier 或在线 Observation。

离线报告至少记录 AP50、AP75、mAP50:95、每类 AP、Precision/Recall、
P50/P95 时延、固定硬件、输入尺寸、阈值、NMS、Batch，以及输入、模型、
类别表和配置的完整 SHA。

## 失败语义

| 故障 | 固定控制路径 | YOLO 评分证据 |
|---|---|---|
| YOLO 空检测 | 当前阶段 VLA 正常继续；令牌规则不变 | 保存合法空预测 |
| YOLO 超时或不可用 | 当前阶段 VLA 正常继续；令牌规则不变 | 记录超时或不可用事件 |
| DetectionPacket 身份或 bbox 非法 | 当前阶段 VLA 正常继续 | 拒绝作为有效预测归档并记录坏包 |
| π0.5 超时或坏响应 | 只重试 Arm_A 固定子任务；耗尽后安全停止 | 同帧 YOLO 证据照常保存 |
| OpenVLA-OFT 超时或坏响应 | 只重试 Arm_B 固定子任务；耗尽后安全停止 | 同帧 YOLO 证据照常保存 |
| 相机、环境或安全故障 | 清空动作队列并安全停止 | 记录关联失败 |

## 后果

优点：

- 四 Agent、多智能体闭环和双臂协作边界固定，团队不再反复变更主线；
- 两个 VLA 都在同一任务中按固定顺序落地，各自训练数据和责任清楚；
- mAP 证据链与机器人动作授权分离，YOLO 漏检会真实反映在 mAP 中；
- 控制事件、检测预测、模型身份和图像身份能够统一追溯；
- 故障归因、重试预算和降级边界清楚。

代价：

- 固定串行牺牲并行吞吐量，以换取双臂安全和 40 天内可交付性；
- 同步 YOLO sidecar 会增加有界调用时延，必须单独报告；
- 必须可靠保存原始预测、空预测、失败事件和版本信息；
- 必须分别报告 YOLO 指标与闭环任务成功率，不能用其中一个替代另一个。

## 验收

- 一次完整任务按 π0.5/Arm_A → 交接核验 → OpenVLA-OFT/Arm_B 的顺序调用两个 VLA；
- 令牌顺序严格为 `A_ONLY → HANDOFF_VERIFY → B_ONLY → NONE`；
- 只有 `robot.arm_a.retreated=true` 且锁臂后三张新鲜观测达到 2/3 复合票数后，
  才依次持久化 `handoff.verified`、`handoff.ready`，并在后者 durable 后授予
  `B_ONLY`；
- 只有 `robot.arm_b.retreated=true` 且料箱位于 `FINISHED_01` 后，任务才成功；
- YOLO 空检测、超时和坏包测试均证明当前阶段 VLA 仍被调用；
- YOLO 故障不改变令牌、不增加 VLA 重试计数，也不改变固定 Agent 职责；
- 日志明确显示 YOLO 当前为同步 sidecar，不宣称真正异步；
- 原始预测包含全部候选框与空帧，可与冻结 GT 重算 mAP；
- 在线 Agent 进程均无法访问 GT。
