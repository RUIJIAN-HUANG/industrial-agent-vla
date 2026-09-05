# 三 Agent 接口契约

## Agent 边界

| Agent | 输入 | 输出 | 禁止职责 |
|---|---|---|---|
| 总控 Supervisor | 用户任务、在线观测、服务响应 | 控制令牌、动作执行命令、审计事件 | 直接加载 VLA/YOLO 模型 |
| YOLO | 一个 CAS 图像引用 | `DetectionPacket` | 规划、控制机械臂、改变令牌 |
| π0.5 | 冻结指令、当前臂图像和 7D 状态 | 统一 `ActionChunk` | 控制未授权机械臂、读取 GT |

## π0.5 请求

`schemas/executor-infer.schema.json` 是唯一机器合同。请求必须使用
`executor: "pi05"`，并在顶层和 `model_input` 中声明一致的 `arm_id`：

```json
{
  "executor": "pi05",
  "arm_id": "Arm_B",
  "model_input": {
    "prompt": "冻结任务指令",
    "arm_id": "Arm_B",
    "observation": {
      "camera": {"full_image": "CAM_B_TOP 的 CAS 引用", "wrist_image": null},
      "robot": {"state": "Arm_B 的 7D 状态", "tcp_pose_m_rad": "Arm_B 的 TCP"}
    }
  }
}
```

总控负责令牌和动作安全；π0.5 只负责在指定 `arm_id` 上返回 7D 动作。
CAS 解析必须经过 `CasRequestImageResolver`，缺图、坏摘要和相机错配一律拒绝。

## YOLO 请求

YOLO 使用 `schemas/perception-detect.schema.json`。它消费不可变 CAS 图像并返回
检测证据；空检测、超时和坏响应只进入 sidecar 证据，不阻塞 π0.5 主控制链路。

## 运行时拓扑

任务请求使用 `task_type: "visual_manipulation"`，由总控映射到冻结任务合同。

```text
Supervisor : task / observation / safety / audit
       ├──> YOLO : /health, /v1/detect, /v1/cancel
       └──> π0.5 : /health, /v1/infer, /v1/cancel
```

第二个 VLA executor 或额外 NLP Agent 配置都不属于当前合同。
