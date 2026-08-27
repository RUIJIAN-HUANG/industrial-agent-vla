# 历史 V1 四 Agent 接口契约（已废除）

版本：2.0
状态：冻结基线
适用范围：Supervisor、π0.5、OpenVLA-OFT、YOLO、Isaac Sim Adapter、离线 mAP Evaluator

> 本文仅供历史接口审计，不得用于正式部署。V2 正式合同见
> `schemas/agent-config-v2.schema.json`、`online-observation-v2.schema.json`、
> `configs/v2-task-profile.json` 与 `src/industrial_agent/v2_supervisor.py`。
>
> 历史场景适用性（2026-08-18）：本文冻结指令、`single_bin_pack_handoff_v1`、
> P01-P04 与自动交接状态机属于 V1 自动闭环兼容基线。当前
> `single_bin_manual_industrial_v2` 人工采集场景继续复用 Canonical Episode、
> 动作维度、图像引用和安全边界等通用合同，但不会反向改写 V1 TaskProfile。

## 1. 冻结边界

本项目只有四个 Agent：

1. `Supervisor Agent`：固定生命周期、FSM、令牌、安全、核验、恢复和遥测；
2. `pi05 Agent`：只控制 `Arm_A`，执行装箱和中央交接；
3. `openvla_oft Agent`：只控制 `Arm_B`，执行同一料箱的协作搬运；
4. `yolo Agent`：只做目标检测、保存检测框和 mAP 证据。

确定性 Safety、Verifier、Isaac Sim Adapter 和离线 mAP Evaluator 不是 Agent。
Supervisor 不做 NLP、任务复杂度判断或模型路由。部署任务指令是冻结预设：

- Arm_A/π0.5：`将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。装箱完成后，将料箱放到中央交接位并返回 HOME_A。失败时重新观察后继续。`
- Arm_B/OpenVLA-OFT：`收到 handoff_ready 后，观察中央交接位，抓稳 Bin_01 并保持水平，将其搬到 FINISHED_01，松开夹爪并返回 HOME_B。`

以上两条是 `single_bin_pack_handoff_v1` 的唯一逐字冻结值，不是任务语义示例。
历史机器真源是 `configs/agent.v1.legacy.json` 中的 `lifecycle.task_profile`，
并由 `schemas/agent-config.schema.json` 的 `const` 约束和
`FixedTaskProfile.validate_frozen()` 双重校验。本文与
`final-frozen-scene-and-flow.md` 必须逐字同步；若需要改写自然语言，必须发布新的
TaskProfile ID 和接口契约版本，不能原地修改 v1。

固定顺序：

```text
A_ONLY
  -> π0.5 / Arm_A
  -> handoff.candidate_checked（仅预检）
  -> HANDOFF_VERIFY
  -> handoff.verified 持久化（锁臂后三帧 2/3）
  -> handoff.ready 持久化（唯一就绪事件）
  -> B_ONLY
  -> OpenVLA-OFT / Arm_B
  -> NONE
```

禁止动态切换两个 VLA，禁止任一 VLA接管另一机械臂。

事件类型统一使用点号风格：

| `event_type` | 固定语义 | 是否允许授予 `B_ONLY` |
|---|---|---|
| `handoff.candidate_checked` | `A_ONLY` 下的单帧候选预检已经完成 | 否 |
| `handoff.verified` | 双臂锁定后，三张新鲜帧的 2/3 复合投票证据已 durable | 否 |
| `handoff.ready` | 已满足全部交接条件，可以让 Arm_B 开始协作 | 是，事件 durable 后 |

`handoff_ready` 只允许作为冻结 Arm_B 自然语言中的业务信号名称；它不是
`event_type`。事件生产者和消费者必须拒绝下划线形式的交接事件类型。

## 2. 通用规则

### 2.1 版本和字段

- JSON 字段统一使用 `snake_case`；
- 所有请求、响应和事件都携带主版本为 `1` 的契约版本；
- 未声明字段一律拒绝，不做静默兼容；
- 布尔值不得冒充整数；
- 浮点值必须有限，不允许 `NaN`、`Infinity`；
- SHA 必须为 `sha256:<64 个十六进制字符>`；
- 服务返回的 ID、版本、Agent 名称和 SHA 必须与请求/部署配置完全一致。

### 2.2 关联 ID

每次在线调用至少关联：

| 字段 | 含义 |
|---|---|
| `trace_id` | Supervisor 生成的单次 run ID |
| `episode_id` | 当前基线与 `trace_id` 相同 |
| `task_id` | 顶层任务或带子任务后缀的任务 ID |
| `subtask_id` | `S01_ARM_A_PACK_HANDOFF` 或 `S02_ARM_B_TRANSPORT` |
| `step_id` | 当前语义子任务内的决策序号 |
| `observation_id` | 当前在线观测唯一 ID |
| `request_id` | 单次 RPC 请求 ID |

`observation_id` 只保证在一个 run 内不重复。跨 run 的帧身份必须使用：

```text
(trace_id, observation_id, camera_id, image_sha256)
```

### 2.3 不可变图像引用

在线 Agent 不接收本地任意路径、HTTP URL、`shm://` 或 `mock://`。图像必须先进入
内容寻址存储，再通过以下格式引用：

```json
{
  "uri": "cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "image_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "camera_id": "CAM_A_TOP",
  "width": 1280,
  "height": 720
}
```

`uri` 中的摘要必须与 `image_sha256` 完全一致。π0.5、OpenVLA-OFT、YOLO
三个模型服务只能从只读 CAS 解析图像；不得跟随重定向或读取任意主机路径。
Isaac Adapter 是唯一在线图像 Producer，必须以读写方式挂载 CAS，并在原子写入
成功后发布 `ImageReference`。

权威实现是 `industrial_agent.image_cas.ImageCas`，完整决策见
[`ADR-0004-shared-image-cas.md`](ADR-0004-shared-image-cas.md)。固定规则为：

- Producer 将 `uint8 H×W×3 RGB` 编码为 PNG；
- `image_sha256` 对 PNG 编码文件的完整字节计算；
- 路径为 `${INDUSTRIAL_AGENT_CAS_ROOT}/sha256/<前两位>/<完整 digest>`；
- Isaac Adapter 原子写入成功后才能发布 `ImageReference`；
- π0.5、OpenVLA-OFT、YOLO 在自身服务入口调用同一个
  `CasRequestImageResolver`，再由它调用底层 `resolve_rgb()`；
- Supervisor 只转发引用，不读取或通过 JSON 传输像素；
- Real 模式解析失败必须返回 CAS/观测错误，禁止使用零图继续推理。

### 2.4 Ground Truth 隔离

在线边界禁止出现以下任意语义及其大小写、连字符或下划线变体：

```text
ground_truth, groundtruth, gt_label, oracle, simulator_truth,
perfect_pose, privileged_state, annotation
```

GT 只允许由离线 Evaluator 读取。YOLO、两个 VLA、Supervisor、在线 Verifier
均不得读取 GT。

## 3. TaskSchema

权威 Schema：`schemas/task.schema.json`

该 Schema 是通用任务信封的结构真源，只校验字段、类型和基础边界；它不会把某个
部署 profile 的自然语言写成通用 `const`。当前部署的逐字值由
`single_bin_pack_handoff_v1` TaskProfile 冻结，并由 Supervisor 在接受任务时执行
第二层校验。因此“通过通用 TaskSchema”不等于“通过当前部署 profile”。

Supervisor 只接受冻结部署任务，不从文本生成路由。示例：

```json
{
  "schema_version": "1.0",
  "task_id": "episode-0001",
  "instruction": "将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。装箱完成后，将料箱放到中央交接位并返回 HOME_A。失败时重新观察后继续。",
  "task_type": "visual_manipulation",
  "postconditions": [
    {
      "kind": "field_equals",
      "path": "task.bin_at_finished",
      "expected": true,
      "min_confidence": 0.6,
      "required_votes": 2
    }
  ],
  "constraints": {},
  "metadata": {}
}
```

要求：

- `instruction` 必须与 Arm_A 冻结指令逐字一致；
- 所有后置条件使用三帧两票；
- `preferred_executor`、`routing`、`switch_executor` 属于废弃字段，必须拒绝；
- TaskSchema 只描述任务，不携带 GT、动作或控制令牌。

## 4. Online Observation

权威 Schema：`schemas/online-observation.schema.json`

### 4.1 必需顶层字段

```text
observation_version
observation_id
timestamp_ms
camera
objects
robot
safety
task
quality
```

四个逻辑引用必须同时存在：

| 键 | 固定 `camera_id` | 用途 |
|---|---|---|
| `camera.full_image` | 冻结 RGB 相机 ID 之一 | 全局留档 |
| `camera.arm_a_rgb` | `CAM_A_TOP` | π0.5 与 A 阶段 YOLO |
| `camera.handoff_rgb` | `CAM_HANDOFF` | 交接核验与 YOLO |
| `camera.arm_b_rgb` | `CAM_B_TOP` | OpenVLA 与 B 阶段 YOLO |

这四个键只引用 **3 台物理相机**：`CAM_A_TOP`、`CAM_HANDOFF`、
`CAM_B_TOP`；`full_image` 是当前阶段已有顶视帧的逻辑引用，不代表第四台相机。
冻结场景没有腕部相机 Prim。统一 VLA 请求中的 `wrist_image` 字段必须存在且值为
JSON `null`；不得用 `full_image` 或其他顶视图伪装腕部视角。

禁止相机回退。例如缺少 `arm_b_rgb` 时，不得改用 `full_image`。

### 4.2 双臂状态

```json
{
  "robot": {
    "active_arm": "Arm_A",
    "arm_a": {
      "tcp_pose_m_rad": [0.45, 0.0, 0.4, 0.0, 0.0, 0.0],
      "state": [0.45, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
      "retreated": false,
      "gripper_open": true,
      "stationary": true
    },
    "arm_b": {
      "tcp_pose_m_rad": [0.40, 0.0, 0.4, 0.0, 0.0, 0.0],
      "state": [0.40, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
      "retreated": true,
      "gripper_open": true,
      "stationary": true
    }
  }
}
```

令牌约束：

| 令牌 | 观测约束 |
|---|---|
| `A_ONLY` | `active_arm` 只能是 `Arm_A/NONE`，Arm_B 必须退避 |
| `HANDOFF_VERIFY` | `active_arm=NONE`，两臂均退避且静止 |
| `B_ONLY` | `active_arm` 只能是 `Arm_B/NONE`，Arm_A 必须退避 |
| `NONE` | `active_arm=NONE`，两臂均静止 |

### 4.3 任务传感事实

允许的任务事实只有：

```text
packed_part_count
bin_at_handoff
bin_at_finished
bin_speed_m_s
arm_a_retreated
arm_b_retreated
status
```

同一帧不得同时声明 `bin_at_handoff=true` 和 `bin_at_finished=true`。
`quality.confidence` 必须存在且为 `[0,1]` 内有限数。

### 4.4 冻结的 `state_7d` 与相机空值

双 VLA 接收的本体状态必须逐项为：

```text
[x_m, y_m, z_m, ax_rad, ay_rad, az_rad, gripper_norm]
```

- 坐标系固定为 `robot_base`；
- `[ax_rad, ay_rad, az_rad]` 是一个 rotation-vector（旋转轴乘旋转角），
  绝对不是 roll/pitch/yaw 欧拉角；
- `tcp_pose_m_rad` 恰好为前 6 项，`state_7d` 的第 7 项只能由控制器确认的
  `gripper_open` 生成；
- 冻结场景只有 `CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP` 三台相机。
  所有 VLA 请求都显式携带 `wrist_image=null`；在线观测缺少该可选键时，
  适配器必须规范化为 `null`，不得创建全黑占位图。

### 4.5 冻结的多频合同

| 层级 | 频率 | 对齐规则 |
|---|---:|---|
| Isaac PhysX | 120Hz | 基础时间栅格 |
| Franka Controller | 60Hz | 每 2 个物理步更新一次控制目标 |
| RGB Render | 30Hz | 每 4 个物理步渲染一帧 |
| VLA Model | 10Hz | 每步 100ms；展开为 6 个控制更新、12 个物理步、3 帧渲染 |

动作块中的每个 7D 增量表示整个 100ms 模型周期的总增量。Isaac 适配器按
60Hz 对平移和 rotation-vector 做分数插值；禁止把完整增量重复执行 6 次。
任何无法同时落在 120Hz 和 60Hz 栅格上的 `duration_ms` 必须在物理写入前拒绝，
不得四舍五入造成跨动作块的相位漂移。

## 5. YOLO Agent

权威 Schema：

- `schemas/perception-health.schema.json`
- `schemas/perception-detect.schema.json`
- `schemas/detection-packet.schema.json`
- `schemas/perception-cancel.schema.json`

### 5.1 输入最小化

YOLO 只能收到 `PerceptionContext`，不得收到完整 TaskSchema 或完整 Observation：

```json
{
  "schema_version": "1.0",
  "request_id": "detect-42",
  "trace_id": "run-1",
  "episode_id": "run-1",
  "task_id": "episode-0001",
  "subtask_id": "S01_ARM_A_PACK_HANDOFF",
  "step_id": 0,
  "observation_id": "obs-42",
  "image": {
    "uri": "cas://sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "image_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "camera_id": "CAM_A_TOP",
    "width": 1280,
    "height": 720
  },
  "allowed_class_names": ["part", "bin"],
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "detector": "yolo",
  "checkpoint_sha": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "class_map_sha": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "config_sha": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
}
```

禁止字段：instruction、prompt、robot、task state、objects、quality、safety、GT。

### 5.2 DetectionPacket

每个 packet 必须回显全部关联 ID、图像身份和三类部署 SHA，并保存：

```text
detection_id
class_id
class_name
confidence
bbox_xyxy
bbox_format=xyxy_pixels
camera_id
image_width
image_height
timing
```

`attributes` 只允许扁平标量值，不允许嵌套字典或数组。零检测是合法预测，
必须保存空 `detections`，不能丢帧。

### 5.3 失败语义

YOLO 是同步采样、失败非门控的评分 sidecar：

- 成功：保存原始 packet；
- 空检测：保存合法空 packet；
- 超时、坏包、服务不可用：保存失败证据，VLA 控制路径继续；
- 硬超时：本次及后续 run 将 YOLO 实例标记为 quarantined，重启隔离服务后才能恢复；
- YOLO 结果不能授予控制令牌，也不能改写 VLA 指令或动作。

### 5.4 COCO/mAP 导出

离线 manifest 中每张图必须包含：

```json
{
  "trace_id": "run-1",
  "observation_id": "obs-42",
  "camera_id": "CAM_A_TOP",
  "image_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "image_id": 42
}
```

导出以 `(trace_id, observation_id, camera_id)` 为键，并再次核验
`image_sha256` 和 `class_map_sha`。这样不同 episode 即使复用 `obs-42` 也不会串帧。
最终由 `scripts/evaluate_detection_map.py` 使用冻结 GT 计算
AP50、AP75、mAP50:95、Precision、Recall 和时延。

## 6. VLA Agent

权威 Schema：`schemas/executor-infer.schema.json`

### 6.1 固定所有权

| Agent | 机械臂 | `full_image` | `wrist_image` | 指令 |
|---|---|---|---|---|
| `pi05` | `Arm_A` | `CAM_A_TOP` | `null` | Arm_A 冻结原始自然语言 |
| `openvla_oft` | `Arm_B` | `CAM_B_TOP` | `null` | Arm_B 冻结协作指令 |

两个 VLA 都在独立进程/容器加载模型，不得在 Supervisor 进程内加载权重。

### 6.2 `/health`

必须返回：

```text
schema_version
service
status=ready
checkpoint_sha
norm_stats_sha
supported_task_types
supported_action_contracts
```

Supervisor 启动时精确核对名称、动作契约、checkpoint SHA 和 norm stats SHA。
任一 VLA 未就绪，不开始 episode。

### 6.3 `/v1/infer`

VLA 请求包含：

- 全部关联 ID；
- 固定 executor 名称和部署 SHA；
- 当前机械臂专属 RGB；
- `wrist_image=null`；当前冻结场景没有腕部相机；
- 当前机械臂状态和 TCP 位姿；
- 当前阶段冻结指令；
- `timeout_ms`。

VLA 不接收另一机械臂的控制目标、YOLO DetectionPacket 或 GT。

### 6.4 Canonical ActionChunk

```json
{
  "contract_version": "1.0",
  "chunk_id": "chunk-0001",
  "task_id": "episode-0001:S01_ARM_A_PACK_HANDOFF",
  "executor": "pi05",
  "action_space": "ee_delta_pose_gripper",
  "frame": "robot_base",
  "translation_unit": "m",
  "rotation_unit": "rad",
  "gripper_unit": "normalized",
  "steps": [
    {
      "values": [0.008, -0.002, 0.004, 0.01, 0.0, -0.01, 1.0],
      "duration_ms": 100
    }
  ]
}
```

动作必须为 7 维有限数，固定语义为
`[dx, dy, dz, dax, day, daz, gripper]`。平移和轴角旋转向量均位于当前机械臂
`robot_base` 坐标系；`[dax, day, daz]` 是一个 rotation-vector，不是三次相互
独立的 world-frame 欧拉角旋转。夹爪 canonical 值允许 `[-1, 1]`，但 Franka
硬件边界统一量化为：`gripper >= 0.5` 表示张开，`gripper < 0.5` 表示闭合。
π0.5 的 `0/1` 与 OpenVLA-OFT 的 `-1/+1` 端点因此具有相同物理含义。

Supervisor 使用滚动时域策略：

1. VLA 可以返回多步；
2. Supervisor 每次只执行第 1 步；
3. 执行后重新观测、YOLO 留证、核验；
4. 丢弃旧 chunk 剩余步骤；
5. 需要继续时，用新帧重新调用同一 VLA。

同一 run 中已经执行过的 `chunk_id` 不得再次执行。

### 6.5 VLA 超时与隔离

- 正常可恢复错误：取消当前请求，使用新观测对同一角色最多重规划一次；
- 硬超时：立即将该 executor 标记为 quarantined，禁止在旧调用未终止时再次推理；
- 硬超时属于执行状态不可信路径，清空动作队列并 safe-stop；
- 恢复路径永远不能从 π0.5 切换到 OpenVLA，反之亦然；
- 生产部署必须使用可取消 transport 或进程隔离；Python daemon 线程只提供
  Supervisor 返回时限，不能杀死底层模型调用。

## 7. Safety 和工作空间

动作验证顺序：

```text
契约版本
-> task_id/executor
-> 7 维/有限数
-> executor-机械臂-令牌绑定
-> 轴限幅
-> 单臂 robot_base 工作空间
-> 对侧机械臂退避互锁
-> 控制器原子 compare-and-execute
```

工作空间按机械臂分别配置，禁止使用一个全局包围盒代替：

```json
{
  "workspace_by_arm": {
    "Arm_A": {
      "frame": "robot_base",
      "min_m": [0.0, -0.60, 0.0],
      "max_m": [0.70, 0.45, 1.0]
    },
    "Arm_B": {
      "frame": "robot_base",
      "min_m": [0.0, -0.25, 0.0],
      "max_m": [0.70, 0.60, 1.0]
    }
  }
}
```

## 8. Isaac Sim / 控制器接口

### 8.1 `observe`

```python
def observe(self) -> Mapping[str, Any]: ...
```

必须返回第 4 节完整在线观测。Supervisor 对调用设置硬 deadline。

### 8.2 `step`

```python
def step(
    self,
    action: ActionStep,
    *,
    arm_id: str,
    control_token: str,
    command_id: str,
    expected_observation_id: str,
    expected_state_digest: str,
) -> Mapping[str, Any]: ...
```

真实控制器必须在同一个 Isaac owner-thread Gate 请求中、写入动作前原子校验：

1. `arm_id` 与 `control_token` 匹配；
2. `command_id` 未执行过；
3. `expected_observation_id` 仍是最新观测；
4. `expected_state_digest` 与 robot/safety/task/objects/quality 当前摘要一致；
5. 对侧机械臂满足退避互锁；
6. Adapter 从 Supervisor 的 authoritative lease source 读取到的当前
   control token 与请求一致；
7. 当前 control lease/stop epoch 仍有效。

校验和写命令必须是一次原子 compare-and-execute。命令采用 fsync 的状态日志：

```text
CLAIMED -> ABORTED                   # 尚未尝试硬件写入
CLAIMED -> APPLIED -> ACKED(result)  # 已写入并保存原始 ACK
```

写控制器前先 durable `CLAIMED`；fsync 可能耗时，因此 claim 后必须再次读取
最新 observation generation、live guard、authoritative lease、stop epoch 和
controller ready，再紧接着写入。`ACKED` 重试只有在请求摘要完全相同时才返回原始
结果，不再次移动机械臂。进程启动发现未决 `CLAIMED/APPLIED` 时，必须立即撤销
lease、safe-stop 并保持 quarantine，禁止猜测动作是否执行。

Supervisor 对 `step` 设置 deadline。超时意味着“动作结果未知”，必须走独立急停通道，
不得假定动作未执行。迟到的旧命令必须因 control lease 已撤销而无法在停机后落地。

### 8.3 `safe_stop`

```python
def safe_stop(self, reason: str) -> SafeStopReceipt: ...
```

返回：

```json
{
  "controller_ack": true,
  "buffers_cleared": true,
  "arm_a_stopped": true,
  "arm_b_stopped": true,
  "stop_epoch": "controller-stop-42"
}
```

只有五项均确认且随后在线观测满足 `active_arm=NONE`、两臂
`stationary=true`，Supervisor 才进入 `SAFE_STOPPED`。否则进入
`SAFE_STOP_FAILED`，不得声称物理停机成功。

`safe_stop` 必须使用独立高优先级通道，不能排在普通 step/infer 队列后。

### 8.4 Isaac Sim 线程 Gate

Omniverse/Isaac stage 和 physics API 可能要求从受控仿真线程调用。B 负责人必须在 D1
完成验证：

- Supervisor Adapter 把 observe/step 请求投递到仿真循环；
- 仿真循环回传有界 ACK；
- watchdog 和 safe-stop 先通过不触碰 Isaac API 的 thread-safe stop Event
  立即撤销 lease，再把 hold/pause/readback 放入独立 urgent 队列；
- urgent ACK 超时后不得取消已经排队的物理 stop；若 owner loop 恢复，仍须执行；
- post-stop observe 仍允许通过 Gate，但任何新 step 必须被 Adapter quarantine；
- 不默认从任意 Python worker thread 直接调用 Isaac API。

生产 standalone 主循环必须持续调用 `gate.pump(max_normal=1)`；Supervisor 运行在
worker 中。若单次 Isaac/PhysX API 本身永久不返回，Python Gate 无法强杀该调用，
此时 receipt 必须保持 unconfirmed、状态进入 `SAFE_STOP_FAILED`，由容器外 watchdog
终止并重启 Adapter，禁止声称物理停机成功。

## 9. 推理后再确认与 TOCTOU

VLA 推理完成后，Supervisor 必须重新采集观测：

- safety、令牌、active arm、对侧机械臂状态或机械臂位姿发生非预期变化：
  立即 safe-stop；
- 物体区域或离散任务事实变化：丢弃旧 chunk，用新观测对同一 VLA 有界重规划；
- `quality.confidence` 等小幅连续噪声不直接触发 safe-stop；
- 默认 `quality/object confidence` 容差为 `0.02`，`bin_speed_m_s` 容差为
  `0.005 m/s`；
- TCP 位姿和机器人状态使用小容差比较；Adapter 默认逐元素容差均为 `1e-3`
  （平移米、旋转弧度、归一化/关节状态各按本字段单位）；
- 最终仍由控制器使用 observation ID 和 state digest 做原子再校验。

`expected_state_digest` 用于证明请求确实基于 Adapter 缓存的原始规划帧；最终
owner-thread 再校验不能对新 telemetry 做字节级 SHA 相等比较，而应按上述离散严格、
连续容差策略比较。超出容差的 robot/safety/active-arm/lease 变化立即 safe-stop；
task/object 离散事实变化则 durable abort 当前未写命令并要求同一 VLA 使用新帧重规划。
跨层只允许通过 `PreWriteStateStaleError` 表达这条可重规划路径；该异常必须同时满足
“尚未尝试硬件写入”与“若已 claim 则 journal 已 durable `ABORTED`”。Supervisor
收到后丢弃 chunk、重新 observe，并消耗同角色一次恢复预算。任何普通异常、超时、
`CLAIMED/APPLIED` 未决状态或写后失败都不得伪装成该异常，必须按执行结果未知
进入独立 safe-stop。

若可重规划错误耗尽预算但整个 run 尚未发生物理写入，Supervisor 可以不调用机械
safe-stop，但必须先把生命周期令牌持久地撤销为 `NONE` 再进入 `FAILED`；任何终态
都不得遗留 `A_ONLY`、`HANDOFF_VERIFY` 或 `B_ONLY` 控制权。

## 10. 交接核验

Arm_A 释放并退避后，Supervisor 先在 `A_ONLY` 下对当前新鲜帧做一次候选预检，
并记录 `handoff.candidate_checked`。该帧只决定是否进入锁臂阶段，不进入最终
投票，也不能授权 Arm_B。候选通过后必须清空旧动作、撤销 A 的运动权限并进入
`HANDOFF_VERIFY`；随后重新采集恰好三个不同 `observation_id`，采用“整帧复合
投票”：

```text
一帧 PASS = 该帧内全部必需条件同时成立
最终 PASS = 3 帧中至少 2 帧复合 PASS
```

禁止从不同帧拼接不同条件形成成功。
候选帧及所有锁臂前帧必须丢弃；最终 2/3 只统计锁臂后新采的三帧，不能把预检
描述成七帧投票。

Arm_A 交接帧必须同时满足：

- 装箱数量为 4；
- 同一 `Bin_01` 位于 `HANDOFF_CENTER`，且不在 `FINISHED_01`；
- 料箱速度 `<= 0.02 m/s`；
- Arm_A 夹爪已释放、已退避；
- Arm_B 已退避；
- 两臂均静止；
- 无急停、保护停和系统故障；
- 质量字段有效。

通过后先将 `handoff.verified` 事件 fsync 到 durable JSONL，再发布并 fsync
唯一就绪事件 `handoff.ready`；只有 `handoff.ready` 的 durable ACK 到达后才允许
`HANDOFF_VERIFY -> B_ONLY`。内存 EventSink 只可用于单元测试，不能作为生产
持久化证明。

最终完成帧必须同时满足：

- 同一 `Bin_01` 位于 `FINISHED_01`，且不在 `HANDOFF_CENTER`；
- 料箱速度 `<= 0.02 m/s`；
- Arm_B 夹爪已释放、已退避；
- Arm_A 已退避；
- 两臂均静止。

## 11. 失败状态

| 类别 | 处理 |
|---|---|
| 无效任务/配置/服务未就绪，且尚未运动 | `FAILED` |
| YOLO 超时/坏包/空检测 | 留失败证据，控制链继续 |
| VLA 可恢复运行错误 | 同一角色、新观测、最多一次重规划 |
| VLA 硬超时 | executor quarantine + safe-stop |
| 动作契约/工作空间/令牌/互锁失败 | safe-stop |
| 急停、保护停、系统故障 | safe-stop |
| controller step 超时 | 执行结果未知 + 独立 safe-stop |
| controller receipt 或停后传感确认失败 | `SAFE_STOP_FAILED` |

## 12. 遥测

生产 EventSink 必须：

- 使用锁保证并发写入和 run 内序号原子性；
- JSONL 每条事件写入后 `flush + fsync`；
- RunResult 只按自身 `run_id` 读取事件，禁止用全局数组切片；
- 不记录 token、密码、base64 图像、GT 或模型隐藏推理；
- 记录 VLA/YOLO 三类 SHA、图像 SHA、动作 chunk ID、command ID、令牌变化、
  核验票数和安全停止 receipt。

## 13. Docker 端口和进程

建议固定：

| 服务 | 端口 |
|---|---:|
| Supervisor | 8000 |
| π0.5 | 8101 |
| OpenVLA-OFT | 8102 |
| YOLO | 8103 |
| Isaac/控制 Adapter | 8200 |

容器停止顺序：

1. Supervisor 撤销 control lease；
2. 通过独立通道请求双臂 safe-stop；
3. 获得 receipt 和停后传感确认；
4. 停止两个 VLA 与 YOLO；
5. 最后停止 Isaac/控制服务。

SIGINT、SIGTERM、`KeyboardInterrupt` 或未处理异常均不得绕过 safe-stop。
真实部署还需容器外 watchdog；进程内异常处理不能替代硬件/控制器急停。

## 14. 负责人验收清单

### A：Supervisor / 集成

- [ ] 固定 FSM 和双 VLA 顺序不可被输入修改；
- [ ] 令牌严格为 `A_ONLY -> HANDOFF_VERIFY -> B_ONLY -> NONE`；
- [ ] VLA 超时 quarantine，不在旧请求存活时重试；
- [ ] 事件按 run 隔离，交接事件 durable；
- [ ] SIGINT/SIGTERM/异常路径均触发急停；
- [ ] 四类 Agent 的服务、版本、SHA 和契约校验齐全。

### B：Isaac Sim / 双臂

- [ ] 完成三相机、双 Franka、四托盘、单料箱和中央交接区；
- [ ] observe/step/safe_stop 满足第 8 节；
- [ ] compare-and-execute、command ID 去重、lease epoch 原子实现；
- [ ] safe-stop 是独立高优先级通道；
- [ ] D1 通过 Isaac 仿真线程 Gate；
- [ ] 两臂从不同时进入共享区。

### C：场景 / 数据

- [ ] 资产命名、单位、坐标系与 canonical manifest 一致；
- [ ] 训练/验证/测试按 episode 分组，禁止相邻帧泄漏；
- [ ] 保存 RGB、动作、状态、指令和失败恢复轨迹；
- [ ] GT 只写离线目录。

### D：OpenVLA-OFT

- [ ] 只接收 Arm_B 指令、相机和状态；
- [ ] `full_image=CAM_B_TOP`，`wrist_image=null`；
- [ ] 自己的 checkpoint/norm stats SHA 可追溯；
- [ ] 输出 canonical ActionChunk；
- [ ] cancel、超时、坏包和 base/tuned 对照可复现。

### E：π0.5

- [ ] 只接收 Arm_A 原始冻结指令、相机和状态；
- [ ] `full_image=CAM_A_TOP`，`wrist_image=null`；
- [ ] LeRobot/openpi 映射、norm stats 与动作单位有测试；
- [ ] 输出 canonical ActionChunk；
- [ ] cancel、超时、坏包和 base/tuned 对照可复现。

### F：YOLO / 核验 / 评测

- [ ] 只接收最小 PerceptionContext；
- [ ] 保存全部候选框、空预测、失败和时延；
- [ ] COCO manifest 使用三元帧键并核验两个 SHA；
- [ ] mAP 可从原始预测和冻结 GT 独立复算；
- [ ] 验证器按整帧复合 2/3 投票，不跨帧拼条件。

## 15. 最小契约测试矩阵

| 用例 | 预期 |
|---|---|
| 两个 VLA health ready 且 SHA 匹配 | 可开始 episode |
| 任一 VLA 未就绪或 SHA 不同 | 运动前 `FAILED` |
| 相机流缺失或 camera ID 错误 | 拒绝观测 |
| CAS URI 与图像 SHA 不一致 | 拒绝观测 |
| YOLO 收到 Task/Robot/GT 字段 | Schema 拒绝 |
| YOLO 空检测 | 保存合法空 packet，VLA 继续 |
| YOLO 硬超时 | 留失败证据并 quarantine，VLA 继续 |
| π0.5 硬超时 | 不重入推理，safe-stop |
| 重复 `chunk_id` 或 `command_id` | 控制器拒绝且 safe-stop |
| 推理期间对侧臂移动 | safe-stop |
| 推理期间物体区域变化 | 丢弃 chunk，同角色有界重规划 |
| step 写后丢 ACK | 状态未知，独立 safe-stop |
| 交接 3 帧中 2 个整帧 PASS | durable handoff 后授予 `B_ONLY` |
| 只有 `handoff.candidate_checked` 或 `handoff.verified` | 不得授予 `B_ONLY` |
| durable `handoff.ready` 到达且交接证据有效 | 才可授予 `B_ONLY` |
| 不同帧分别满足不同条件 | 不得通过 |
| safe-stop receipt 缺项 | `SAFE_STOP_FAILED` |
| receipt 成功但传感器仍在运动 | `SAFE_STOP_FAILED` |
| 两个 run 共享 EventSink 并发 | RunResult 事件零串入 |
| 两个 run 复用同 observation ID | mAP manifest 仍按 trace/camera 正确绑定 |
