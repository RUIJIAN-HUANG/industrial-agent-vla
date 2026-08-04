# Isaac Sim 5.1 执行 Adapter

本文说明双 Franka 场景的真实控制器执行边界。它不是完整比赛闭环的替代品：
生产闭环仍必须接入三路 RGB RenderProduct、CAS 写入/解析、Supervisor、双 VLA、
YOLO 与在线 verifier。本 Adapter 只负责把已经通过 Supervisor 安全校验的 canonical
7-D 动作，经受控线程 Gate 送入 Isaac Sim 5.1，并提供可审计的停止与去重语义。

## 1. 组件与职责

| 文件 | 职责 |
|---|---|
| `src/industrial_agent/isaac_runtime.py` | Supervisor worker 与 Isaac owner thread 之间的有界请求 Gate；普通 FIFO 和 urgent stop FIFO 分离 |
| `src/industrial_agent/isaac_environment.py` | observation/lease/stop epoch 的 compare-and-execute、容差复查、durable command journal |
| `simulation/isaac_franka_controller.py` | Isaac Sim 5.1 Lula IK、Franka articulation 写入、双臂 hold/pause/readback |
| `simulation/run_isaac_adapter_smoke.py` | 不使用相机证据的 controller-only smoke；验证 Gate、一个微动作和 safe-stop |

## 2. 线程模型

Supervisor 的 hard deadline 会从临时 worker thread 调用 `observe/step/safe_stop`；
Isaac stage、articulation 和 physics API 则只允许在创建 World/Controller 的 owner
thread 执行。因此生产入口必须采用下列结构：

```text
Supervisor worker
    │  environment.observe/step/safe_stop
    ▼
IsaacMainThreadGate
    ├─ normal FIFO: observe、atomic compare-and-execute
    └─ urgent FIFO: hold、pause、stop readback
            │
            ▼
Isaac owner loop: gate.pump(max_normal=1)
```

普通 `gate.call()` 必须从 Supervisor worker 发起；若在 owner thread 直接调用会
明确失败，因为内联执行无法兑现有界 ACK。`call_stop()` 是唯一允许 owner thread
直接触发的例外，它仍先撤销控制权，再执行 hold/pause/readback。

急停分两阶段：

1. 调用线程立即执行 `controller.request_stop()`；它只设置 `threading.Event` 和
   单调 stop epoch，不调用任何 Isaac API；
2. owner thread 从 urgent FIFO 执行 `confirm_safe_stop()`，对两臂分别写入 hold、
   清零速度、暂停 world，并读取 controller/velocity 状态。

如果某一次 `world.step()` 或其他 Isaac API 永久不返回，Gate 无法在 Python 进程内
强杀该调用。此时 stop receipt 必须是 unconfirmed，Supervisor 进入
`SAFE_STOP_FAILED`；容器外 watchdog 必须终止并重启 Adapter。严禁把“已发送 stop
Event”描述成“机械臂已经停止”。

## 3. 原子执行顺序

`step()` 在一个 normal Gate operation 内执行：

```text
静态 arm/token/action 校验
→ expected observation ID + 原始 planning digest
→ owner-thread live guard（离散严格、连续容差）
→ authoritative lifecycle token + opposite-arm retreat
→ controller ready
→ fsync CLAIMED
→ 再读 latest observation generation/live guard/token/stop epoch/ready
→ controller.execute_action
→ fsync APPLIED
→ 新鲜 post-action observation
→ fsync ACKED + 原始 result
```

Journal 状态：

```text
CLAIMED -> ABORTED
CLAIMED -> APPLIED -> ACKED(result)
```

- `ACKED` 同请求重试返回原 result，不再次执行；
- 同一 `command_id` 对应不同请求摘要会被拒绝；
- 启动时发现 `CLAIMED/APPLIED` 表示物理结果未知，Adapter 会立即 safe-stop 并永久
  quarantine；必须人工核对场景和 journal 后再建立新实例；
- journal 使用 JSONL、file fsync；Linux 首次创建还会 fsync 父目录。

只有 task/object/quality 在写入前发生漂移，且 journal 已证明命令未写入时，
Adapter 才抛出 `PreWriteStateStaleError`。Supervisor 会丢弃旧 chunk，并让同一个
VLA 使用新观测做一次有界重规划。任何写后失败或结果未知都不会走此恢复分支。

## 4. canonical 动作

```text
[dx, dy, dz, dax, day, daz, gripper]
```

- 平移：米，`robot_base`；
- 旋转：弧度，`[dax, day, daz]` 是 rotation-vector（轴角向量）；
- 夹爪：canonical 范围 `[-1, 1]`；硬件边界 `>= 0.5` 张开，`< 0.5` 闭合；
- Controller 每个 physics tick 都检查 stop Event；
- Lula base pose 在每次求解前刷新，TCP target 再从 robot_base 转到 world。

Lula IK 本身不提供双臂/环境碰撞规划。本 Adapter 仍依赖冻结 workspace、单臂令牌、
对侧退避互锁和小步滚动时域；它不能被宣传为 collision-aware planner。

### 4.1 120/60/30/10Hz 多频同步

`src/industrial_agent/sync_contract.py` 是频率真源：PhysX 120Hz、控制器
60Hz、RGB 渲染 30Hz、VLA 推理/动作采样 10Hz。一个 100ms 模型动作必须严格
展开为 6 个控制目标、12 个物理步和 3 次渲染。控制目标使用同一初始 TCP 的
笛卡尔分数插值，旋转部分始终缩放同一个 robot_base rotation-vector；因此最终
增量只执行一次，不会被 6 倍放大。渲染按全局物理 tick 降采样，跨 ActionChunk
保持相位连续。

`duration_ms` 必须同时落在 120Hz 物理栅格和 60Hz 控制栅格上，否则 Controller
在任何 IK/关节写入前 fail-closed。标准双 VLA 响应固定为每步 100ms；250ms 等
控制器 smoke 诊断时长仍可使用，只要满足整数栅格。

## 5. Supervisor 接线要求

构造环境时必须显式提供：

```python
environment = IsaacExecutionEnvironment(
    observation_source=capture_complete_online_observation,
    state_guard_source=capture_guarded_state,
    control_lease_source=lambda: supervisor.current_control_token,
    controller=controller,
    runtime_gate=gate,
    command_ledger_path="artifacts/isaac-adapter/command-journal.jsonl",
    runtime_observe_timeout_s=1.0,  # 短于 post-stop observe 的 2 s deadline
    runtime_action_timeout_s=10.0,  # 短于 Supervisor 15 s action deadline
    runtime_stop_timeout_s=1.0,   # 必须短于 Supervisor safe_stop deadline
)
```

`observation_source` 必须生成符合 `schemas/online-observation.schema.json` 的完整在线
观测和真实 CAS URI；`state_guard_source` 必须从同一实时传感/控制状态构造
robot/safety/task/objects/quality，不能使用 simulator GT 坐标。所有会触碰 Isaac 的
source 都只能由 Gate operation 调用。

## 6. 本地契约测试

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_isaac_runtime.py \
  tests/test_isaac_environment.py \
  tests/test_isaac_franka_controller.py
```

这些测试覆盖 owner-thread 投递、有界超时、排队请求取消、urgent stop、并发 step、
stop 后迟到结果、实时状态在 fsync 期间变化、observation generation、current lease、
durable ACK、未决命令重启和逐臂停止容错。

## 7. Linux Isaac Sim 5.1 smoke

先在 GUI 中分别验证两臂。每个失败实验保留独立 journal 和 result，不能删除未决
journal 后直接继续动作。

```bash
cd "$HOME/Sceneconstruction/industrial-agent-vla"
export ISAAC_SIM_ROOT="$HOME/isaacsim"

"$ISAAC_SIM_ROOT/python.sh" simulation/run_isaac_adapter_smoke.py \
  --arm-id Arm_A \
  --command-ledger artifacts/isaac-adapter/arm-a-command-journal.jsonl \
  --result-file artifacts/isaac-adapter/arm-a-smoke.json

"$ISAAC_SIM_ROOT/python.sh" simulation/run_isaac_adapter_smoke.py \
  --arm-id Arm_B \
  --command-ledger artifacts/isaac-adapter/arm-b-command-journal.jsonl \
  --result-file artifacts/isaac-adapter/arm-b-smoke.json
```

GUI 两臂都通过后，分别加 `--headless` 重跑。PASS 结果至少包含：

```json
{
  "status": "PASS",
  "arm_id": "Arm_A",
  "arm_joint_delta_norm": 0.01,
  "finger_delta_norm": 0.0,
  "tcp_delta_norm_m": 0.005,
  "tcp_delta_base_m": [0.0, 0.0, 0.005],
  "safe_stop_confirmed": true
}
```

脚本会验证：

- 七个 arm joints 的变化不是由 finger movement 冒充；
- TCP 位移量和 robot-base Z 方向与请求在容差内一致；
- stop receipt 五项全部确认；
- worker 的 observe/state guard/action/stop 真实经过 owner-thread Gate。

## 8. 本 PR 没有证明的内容

即使两次 smoke PASS，也只能说明执行 Adapter 的最小路径成立，不能说明完整比赛闭环
已经完成。下列证据仍需由后续生产入口和 Isaac 5.1 实机/仿真运行提供：

- 三路 1280×720 RGB RenderProduct 和 CAS bridge；
- 完整 ObservationGateway schema 验证；
- Supervisor + π0.5 + OpenVLA-OFT + YOLO 的真实 Docker 联调；
- 双臂 IK 可达性、碰撞、抓取稳定性与 50-seed 结果；
- 外部 process watchdog 的 kill/restart 演练；
- base/tuned 双 VLA 对照和 YOLO mAP 证据。

因此，仓库 CI 通过代表“纯 Python 契约和并发回归通过”，不代表“Isaac 现场验收通过”。
