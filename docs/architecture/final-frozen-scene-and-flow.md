# 最终冻结场景与双 VLA 串行闭环

> 状态：最终冻结
>
> 日期：2026-07-26
>
> 适用范围：Isaac Sim MVP、Docker 演示、技术报告和答辩视频

## 1. 场景边界

![单料箱双臂仿真平台](assets/isaac-sim-single-bin-annotated-platform-v1.png)

场景名称：**双臂单料箱装箱—静态交接—成品搬运工作站**。

- 两台 Franka 固定在同一工作台，当前都使用二指夹爪；
- Arm_A 负责左侧装箱区，Arm_B 负责右侧成品区；
- 一个 2×3 料箱从 `PACK_STATION` 出发；
- 四个红色零件按 A/B/C/D 区数量 `2/1/1/0` 放置，其中 P02 倒放；
- Arm_A 装完四件后把料箱直接放到固定 `HANDOFF_CENTER`；
- 没有传送带、移动交接台、第二个料箱或叠箱动作；
- Arm_A 退出后，Supervisor 完成三帧核验，再允许 Arm_B 抓取；
- Arm_B 把同一料箱搬到唯一成品位 `FINISHED_01`。

## 2. 世界坐标与节点

坐标单位为米，`+X` 从 Arm_A 指向 Arm_B，`+Y` 指向工作台后侧，`+Z`
竖直向上。配置真源是
[`../../simulation/configs/single_bin_scene_v1.json`](../../simulation/configs/single_bin_scene_v1.json)。

| 对象 | Isaac Sim Prim | 初始位置 `(x,y,z)` |
|---|---|---:|
| Arm_A / π0.5 | `/World/Robots/Arm_A` | `(-0.55,-0.30,0.750)` |
| Arm_B / OpenVLA | `/World/Robots/Arm_B` | `(+0.50,-0.30,0.750)` |
| 空料箱 | `/World/Bins/Bin_01` | `(-0.35,-0.15,0.785)` |
| P01 正放 | `/World/Parts/P01` | `(-0.90,+0.20,0.772)` |
| P02 倒放 | `/World/Parts/P02` | `(-0.80,+0.20,0.772)` |
| P03 正放 | `/World/Parts/P03` | `(-0.60,+0.20,0.772)` |
| P04 正放 | `/World/Parts/P04` | `(-0.85,0.00,0.772)` |
| 静态交接位 | `/World/Stations/HANDOFF_CENTER` | `(0.00,0.00,0.785)` |
| 唯一成品位 | `/World/Stations/FINISHED_01` | `(+0.70,+0.10,0.785)` |

关键平面距离均不超过 MVP 软限制 `0.65 m`；这只是几何预检，最终还必须在
Isaac Sim 中验证预抓取、抓取、抬升、放置 IK、碰撞和关节余量。

## 3. RGB 相机

| 相机 | Prim | 建议位置 | 主要用途 |
|---|---|---:|---|
| A 区顶视 RGB | `/World/Cameras/CAM_A_TOP` | `(-0.65,-0.15,1.50)` | P01–P04、格口、Arm_A 装箱 |
| 交接 RGB | `/World/Cameras/CAM_HANDOFF` | `(0.00,-0.45,1.18)` | 箱体进入 ROI、稳定、夹爪释放 |
| B 区顶视 RGB | `/World/Cameras/CAM_B_TOP` | `(+0.60,-0.18,1.45)` | Arm_B 抓箱、成品位放置 |

冻结 MVP **恰好只有以上三台物理 RGB 相机**，没有腕部相机，也不存在
`CAM_A_WRIST` 或 `CAM_B_WRIST` Prim。VLA 服务请求为保持统一接口仍包含
`wrist_image` 字段，但在本 TaskProfile 中必须逐次传 JSON `null`；不得从其他顶视
相机伪造或回退该字段。未来增加腕部相机必须发布新的场景和 TaskProfile 版本。

VLA 接收未叠加检测框的完整 RGB 图像。YOLO 使用同一原始帧的不可变副本，
以 `observation_id + image_sha256` 与动作和事件关联。

## 4. 固定角色指令

以下两条是 `single_bin_pack_handoff_v1` 的唯一逐字冻结值，不是任务语义示例。
机器可执行真源与版本变更规则见
[`interface-contracts.md`](interface-contracts.md#1-冻结边界)。

### π0.5 / Arm_A

```text
将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。装箱完成后，将料箱放到中央交接位并返回 HOME_A。失败时重新观察后继续。
```

### OpenVLA-OFT / Arm_B

```text
收到 handoff_ready 后，观察中央交接位，抓稳 Bin_01 并保持水平，将其搬到 FINISHED_01，松开夹爪并返回 HOME_B。
```

两条自然语言均在任务配置中预设。Supervisor 只原样传输，不识别关键词、
不判断任务难度、不生成新子指令。

## 5. 完整闭环例子

下表第二列是便于答辩和数据标注的**业务操作阶段**，不是
`industrial_agent.fsm.AgentState` 枚举。监控、事件 `state` 字段和代码分支只能
使用下列真实 AgentState；不得把 `OBSERVE_A`、`INFER_A`、`HANDOFF_READY` 等
业务名称当作 FSM 状态。

| 业务操作阶段 | 对应真实 `AgentState` |
|---|---|
| 初始化、Reset、任务校验 | `IDLE → VALIDATING_TASK` |
| 生成固定双子任务计划 | `PLANNING` |
| `OBSERVE_A`、`OBSERVE_B` | `OBSERVING`；YOLO 留证时短暂进入 `PERCEIVING` |
| `INFER_A`、`INFER_B` | `ASSIGNING_ROLE → EXECUTING` |
| `EXECUTE_A`、`EXECUTE_B` | `EXECUTING` |
| `VERIFY_A`、`HANDOFF_VERIFY`、`FINISHED_VERIFY` | `VERIFYING` |
| 切换子任务或继续闭环 | `ADVANCING_SUBTASK → OBSERVING` |
| 同角色有界恢复 | `REPLANNING → OBSERVING` |
| 成功或终止 | `SUCCEEDED`、`FAILED`、`SAFE_STOPPED`、`SAFE_STOP_FAILED` |

| 步骤 | 业务操作阶段（非 FSM 枚举） | 令牌 | 执行者 | 动作与证据 | 通过条件 |
|---:|---|---|---|---|---|
| 1 | `IDLE → RESET` | `NONE` | Supervisor、Isaac | 双臂归零、清动作队列、确认交接位为空 | `HOME_A && HOME_B` |
| 2 | `RESET → OBSERVE_A` | `A_ONLY` | Supervisor、相机 | 获取 A 区新鲜 RGB；同步调用 YOLO 保存 P01–P04 bbox，失败只留证 | 图像与机器人状态有效 |
| 3 | `OBSERVE_A → INFER_A` | `A_ONLY` | π0.5 | 接收原始 A 指令、A 区全图和 Arm_A 状态 | 返回 Arm_A ActionChunk |
| 4 | `INFER_A → EXECUTE_A` | `A_ONLY` | Safety、Arm_A | 只执行首个安全动作，靠近 P01 | 无越界、无碰撞 |
| 5 | `EXECUTE_A → VERIFY_A` | `A_ONLY` | Supervisor、相机 | 清空旧动作，重新观察；YOLO 留存新帧预测 | 抓取/放置证据有效 |
| 6 | `VERIFY_A ↔ INFER_A` | `A_ONLY` | π0.5、Arm_A | 以新观测继续抓 P01 并放入格口；对 P02 先纠姿，再处理 P03/P04 | 四件均在箱内且无掉落 |
| 7 | `VERIFY_A → PACK_COMPLETE` | `A_ONLY` | Supervisor | 核对四个零件、料箱身份和 Arm_B 仍在 HOME_B | `pack_complete` |
| 8 | `PACK_COMPLETE → ARM_A_HANDOFF` | `A_ONLY` | π0.5、Arm_A | 重新观察后抓料箱，将其放到固定中央 ROI | 箱体已释放 |
| 9 | `ARM_A_HANDOFF → ARM_A_RETREAT` | `A_ONLY` | Arm_A | Arm_A 返回 `HOME_A`，清空控制队列 | Arm_A 离开共享区 |
| 10 | `ARM_A_RETREAT → HANDOFF_CANDIDATE` | `A_ONLY`，通过后切 `HANDOFF_VERIFY` | Supervisor | 用当前新鲜帧做一次候选预检并记录 `handoff.candidate_checked`；预检通过后撤销 A 令牌、清空队列并锁定双臂 | 候选通过不代表 ready；双臂已禁止进入共享区 |
| 11 | `HANDOFF_VERIFY` | `HANDOFF_VERIFY` | Supervisor、相机、Verifier | 锁臂后重新采集恰好 3 张不同帧；核对箱体在 ROI、速度低、A 已释放并退回；2/3 通过后持久化 `handoff.verified` | 至少 2 帧复合 PASS，候选帧不计票 |
| 12 | `HANDOFF_VERIFY → HANDOFF_READY` | `HANDOFF_VERIFY → B_ONLY` | Supervisor | `handoff.verified` 已 durable 后，再持久化唯一就绪事件 `handoff.ready`；收到 durable ACK 后才授予 B 令牌 | 两个事件顺序可审计，B 此前零动作 |
| 13 | `HANDOFF_READY → OBSERVE_B` | `B_ONLY` | 相机、YOLO | 获取 B/交接区新鲜 RGB；同步调用 YOLO 保存满箱 bbox，失败只留证 | 图像与 Arm_B 状态有效 |
| 14 | `OBSERVE_B → INFER_B` | `B_ONLY` | OpenVLA-OFT | 接收固定 B 指令、完整图像与 Arm_B 状态 | 返回 Arm_B ActionChunk |
| 15 | `INFER_B ↔ EXECUTE_B` | `B_ONLY` | Safety、Arm_B | 单步滚动抓稳料箱、抬升、保持水平、移动到成品位 | 每步后均有新观测 |
| 16 | `EXECUTE_B → ARM_B_RETREAT` | `B_ONLY` | Arm_B、Supervisor | 同一料箱已进入 `FINISHED_01` 且夹爪释放；清空旧动作后，Arm_B 返回 `HOME_B` | Arm_B 已退避，Arm_A 仍退避 |
| 17 | `ARM_B_RETREAT → FINISHED_VERIFY` | `B_ONLY` | Supervisor、相机、Verifier | 重新采集三张不同新鲜帧，核对同一料箱、完成区、料箱速度、夹爪释放、双臂退避与静止 | 三帧中至少两帧整帧复合 PASS |
| 18 | `FINISHED_VERIFY → TASK_SUCCEEDED` | `B_ONLY → NONE` | Supervisor | 后置条件通过后撤销 B 令牌并持久化最终结果 | `active_arm=NONE`，双臂静止，成品到位 |
| 19 | `TASK_SUCCEEDED → IDLE` | `NONE` | Supervisor、离线评测 | 保存 Trace、动作、`agent-events.jsonl`、`yolo-evidence.jsonl`、视频；离线导出 `raw_predictions.json` 并生成 `detection_metrics.json` | 结果包完整且可复算 |

## 6. 当前阶段恢复

```text
动作失败
→ 立即清空当前机械臂的旧动作
→ 获取该工作区的新 Observation
→ 再次调用当前阶段的固定 VLA
├─ 预算内恢复：继续
└─ 预算耗尽
   ├─ 整个 run 尚未发生物理写入：持久撤销令牌为 NONE → FAILED
   └─ 已发生物理写入或执行结果未知：独立 safe-stop
      ├─ 停机和停后传感确认成功：SAFE_STOPPED
      └─ 回执或停后确认失败：SAFE_STOP_FAILED
```

- Arm_A 阶段只调用 π0.5；
- Arm_B 阶段只调用 OpenVLA-OFT；
- 交接核验失败时两臂都不动；
- 相机、控制器、急停或保护停故障直接安全停止；
- YOLO 旁路失败必须留证，但不会改变两个 VLA 的固定角色。

## 7. 成功判定

只有以下条件全部成立，任务才是成功：

1. P01–P04 均进入同一 `Bin_01`，P02 已正向，箱外无掉落；
2. Arm_A 把 `Bin_01` 放到固定 `HANDOFF_CENTER` 并返回 `HOME_A`；
3. 三张不同帧中至少两票确认交接条件；
4. 令牌顺序严格为 `A_ONLY → HANDOFF_VERIFY → B_ONLY → NONE`；
5. Arm_B 搬运的是同一 `Bin_01`，最终位于 `FINISHED_01`；
6. Arm_B 返回 `HOME_B`，两臂从未同时进入共享区；
7. 每个物理动作后都有新 observation 和重新推理/核验；
8. 在线 `agent-events.jsonl`、`yolo-evidence.jsonl`、`trace.json`、视频，以及离线
   `raw_predictions.json`、`detection_metrics.json` 均可通过 `trace_id` 和图像 SHA
   关联且可复算。

交接事件类型唯一采用点号风格。候选预检事件
`handoff.candidate_checked` 在进入交接的运行中至少出现一次，并可因重试出现
1..N 次；不可逆里程碑顺序固定为
`handoff.verified → handoff.ready`。候选预检和 `handoff.verified` 只表示
预检或核验证据，**均不表示 Arm_B 已获准动作**；只有 durable
`handoff.ready` 才是授予 `B_ONLY` 的就绪事件。冻结 Arm_B 自然语言中的
`handoff_ready` 是业务信号名称，不是 `event_type`。
