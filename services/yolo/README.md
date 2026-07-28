# YOLO 感知 Agent 服务

负责人：F

当前状态：合同与客户端骨架已实现；生产模型运行时和权重不在仓库中。

本目录是第四个 Agent 的独立部署边界，也是同步调用、失败非门控的评分
sidecar。服务每次只检测一张不可变相机帧，不调用 π0.5 或 OpenVLA-OFT，
不发放双臂令牌，也不决定 Supervisor 生命周期。在线进程禁止读取离线标注、
目标真实位姿、抓取点或其他特权信息。

## 当前调用语义

Supervisor 对选定的新鲜观测同步调用一次 `/v1/detect`，并等待成功、明确错误或
有界超时后继续当前固定阶段。因此当前实现不是异步队列或真正并行推理，必须把
YOLO 调用时延计入端到端日志。

同步调用不等于硬门：

- 合法空检测、超时、服务不可用或坏响应只记录 sidecar 证据；
- YOLO 成功不是 π0.5 或 OpenVLA-OFT 请求的必填条件；
- YOLO 成功不是 `A_ONLY → HANDOFF_VERIFY → B_ONLY` 的令牌条件；
- YOLO 故障不消耗 VLA 重试预算，也不改变两个 VLA 的固定职责；
- Supervisor 仍依据冻结 FSM、新鲜在线观测、机器人遥测和安全条件继续、
  重试当前固定子任务或安全停止。

## 端点

| 端点 | 用途 | 合同 |
|---|---|---|
| `GET /health` | 返回模型、类别表、配置和合同身份 | `schemas/perception-health.schema.json` |
| `POST /v1/detect` | 对一个 `observation_id` 和图像摘要返回检测结果 | `schemas/perception-detect.schema.json` |
| `POST /v1/cancel` | 取消排队推理并清理服务上下文 | `schemas/perception-cancel.schema.json` |

成功的 `/v1/detect` 响应内嵌
`schemas/detection-packet.schema.json`。响应必须原样回显所有关联字段。
Supervisor 客户端会拒绝不一致的 `observation_id`、`image_sha256`、相机身份、
图像尺寸、checkpoint、类别表摘要、配置摘要、任务、步骤或 trace。

## 运行时职责

1. 只允许通过完整 `sha256:<64 位十六进制>` 身份加载 checkpoint。
2. 按 `/health` 暴露的摘要加载冻结类别表和推理配置。
3. 解析请求中的图像 URI，用 `image_sha256` 校验字节，只执行一次预处理、
   模型推理和 NMS。
4. 输出像素坐标系 `xyxy` 检测框，并记录预处理、推理、NMS 和端到端时延。
5. 保存全部候选检测与合法空预测，不能只保留一个目标框。
6. 用 `trace_id + observation_id + image_sha256` 关联控制 trace。
7. 在线服务进程不得读取离线评测标注目录。

第 3 步必须通过
`industrial_agent.service_images.CasRequestImageResolver.resolve_yolo_request()`
实现，并与两个 VLA 共用 `INDUSTRIAL_AGENT_CAS_ROOT` 只读卷。YOLO 不得使用
HTTP 下载、任意文件路径或自行维护另一套 CAS resolver。

仓库已提供 [`handler.py`](handler.py) 的 `build_v1_detect_handler()` 作为
`POST /v1/detect` 强制入口核心：它先解析并校验 CAS，再用只读 RGB 数组替换
请求中的 URI 引用后调用注入的 YOLO backend。HTTP 外壳不得绕过该 handler。

## 失败与数据标签

空 `detections` 是合法成功响应。服务错误使用
`schemas/perception-detect.schema.json` 中的 `error.code`、`error.message` 和
`error.retryable`，不引用额外的全局错误枚举。

在线错误码用于控制 trace；采集数据需要诊断标签时，由 C/F 在隔离的数据处理
阶段映射到 `dataset_failure_label`。该标签不回注在线 Observation，也不参与
令牌判定。

| 情况 | 在线控制影响 | 评分证据 |
|---|---|---|
| 空检测 | 无；当前固定 VLA 继续 | 保存合法空预测 |
| 超时或服务不可用 | 无；当前固定 VLA 继续 | 保存关联失败事件 |
| 身份摘要或同帧关联不一致 | 无；当前固定 VLA 继续 | 拒绝作为有效预测归档 |
| bbox 越界、倒置或零面积 | 无；当前固定 VLA 继续 | 记录坏包并进入离线 QA |

## 离线 mAP

在线 YOLO 只产生原始预测。离线评测器把归档预测与冻结 COCO GT 结合，计算
AP50、AP75、mAP50:95、Precision/Recall、每类指标和时延统计。GT 目录不得
挂载到 Supervisor、YOLO、π0.5、OpenVLA-OFT 或在线 Verifier 容器。

## Python 客户端与测试替身

`industrial_agent.perception.YoloHTTPAdapter` 是严格的跨进程客户端。它支持注入
transport，使 HTTP 依赖留在 Supervisor 核心之外。
`MockPerceptionAgent` 实现同一个 `PerceptionAgent` 协议，用于编排、
同帧关联和故障注入测试。

不要向本目录提交模型权重、数据集、缓存、凭据或本机绝对路径。
