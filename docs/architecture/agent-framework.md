# 当前三 Agent 架构

当前运行时只有三个 Agent：

1. 总控 Supervisor：校验任务、管理 FSM/安全令牌、调用服务、验证后置条件和安全停止。
2. YOLO：独立感知 sidecar，只输出检测证据，不规划、不发放控制权。
3. π0.5：唯一 VLA，通过请求中的 `arm_id` 控制 `Arm_A` 或 `Arm_B`。

`Arm_A` 和 `Arm_B` 是执行机构，不是 Agent。π0.5 服务可以串行服务两臂，
不能在同一控制令牌下同时驱动两臂。

```text
用户任务
   ↓
总控 Supervisor ──同步旁路──> YOLO 检测证据
   │
   └──/v1/infer + arm_id──> π0.5 ──> Arm_A 或 Arm_B
```

生产 Compose 只启动 `pi05` 和 `yolo` 两个模型服务；总控运行在主进程/仿真入口中。
因此按 Agent 计数是 3，按容器计数是 2 个模型服务加 1 个总控运行时。

## 固定不变量

- 执行器白名单只有 `pi05`；不存在第二个 VLA 适配器或服务。
- π0.5 请求和 `model_input` 必须同时携带相同的 `arm_id`。
- `Arm_A` 使用 `CAM_A_TOP`，交接核验使用 `CAM_HANDOFF`，`Arm_B` 使用 `CAM_B_TOP`。
- 每次动作后重新获取观测；失败只在同一个 π0.5 服务内有界重试。
- YOLO 失败不改变 π0.5 的控制路径，也不能获得机械臂控制权。
- 生产配置必须只有 `executors.pi05`；新增模型服务必须先改变架构决策。

权威合同位于 `schemas/`，生产装配入口为
`src/industrial_agent/supervisor_main.py`。
