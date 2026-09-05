# JSON Schema 索引

| 文件 | 责任方 | 用途 |
|---|---|---|
| `agent-config-v2.schema.json` | A/E | V2 正式总控配置 |
| `task.schema.json` | A | 用户任务输入 |
| `task-plan.schema.json` | A | 语义 TaskPlan |
| `online-observation-v2.schema.json` | A/B/E/F | V2 正式在线传感观测与终局证据；禁止 GT |
| `online-observation-common.schema.json` | A/B/E/F | V2 相机、双臂状态、安全与质量公共定义 |
| `online-observation.schema.json` | B/C/F | 已废除 V1 在线观测，仅历史回归 |
| `perception-health.schema.json` | A/F | YOLO 身份、健康及三类部署 SHA |
| `perception-detect.schema.json` | A/F | 同帧 YOLO detect 请求/响应信封 |
| `detection-packet.schema.json` | A/F/离线评测 | bbox、类别、图像关联和检测时延；不进入 VLA |
| `perception-cancel.schema.json` | A/F | YOLO 幂等取消 |
| `executor-health.schema.json` | D/E | VLA 服务身份与健康 |
| `executor-infer.schema.json` | A/D/E | 推理请求/响应信封 |
| `executor-cancel.schema.json` | A/D/E | 幂等取消 |
| `action-chunk.schema.json` | A/B/D/E | 统一 N×7 动作 |
| `canonical-episode.schema.json` | A/C + B/D/E/F 评审 | V1 自动闭环 HDF5 Episode；允许显式 mask padding |
| `canonical-episode-v2.schema.json` | A/C + B/E 评审 | V2 P01→S11 人工采集；严格 7D、无 padding、固定场景/任务/指令 |
| `verify.schema.json` | A/F | 在线后置条件核验 |
| `event.schema.json` | A/F | 结构化事件证据 |

CI 会用 Draft 2020-12 元 Schema 校验所有文件，并验证默认 Agent 配置。
Schema 变化必须先更新接口文档和契约测试，再修改生产服务。

Canonical 1.0 只为历史数据审计保留，正式训练仅使用 V2。1.0 顶层版本键为
`schema_version="1.0"`；V2 顶层版本键为
`canonical_schema_version="2.0"`。Recorder、Reader 和转换器必须按版本显式选择，
不得通过改写字段把 1.0 Episode 冒充为 V2。

Schema 只覆盖跨进程、落盘或对外交换的 JSON 合同。`RunMemory`、
`StateTransition`、`ExecutorDescriptor`、`ExecutionContext`、
`VerificationResult`、`ConditionResult` 等进程内类型不单独建立 Schema；
它们只有在转换为上表中的事件、推理信封或核验结果后才允许跨边界。
`CocoExportManifest` 由离线导出代码的严格 `from_dict()`/`to_dict()` 合同校验，
不属于在线 Agent 接口。

所有 `checkpoint_sha` 与 `norm_stats_sha` 必须使用完整
`sha256:<64 位十六进制>`；配置文件仅允许明确的待替换占位符，任何真实服务
启动时都会拒绝该占位符以及 `latest`、版本昵称和缩写摘要。

YOLO 的 `checkpoint_sha`、`class_map_sha`、`config_sha` 与 `image_sha256` 也使用
同样的完整摘要。DetectionPacket 必须和 VLA 输入共享同一 `trace_id`、
`observation_id` 与 `image_sha256`。所有在线 Schema 都禁止 GT、annotation、
oracle、真实目标位姿和抓取点；mAP 的冻结 GT 不属于在线 Schema。
