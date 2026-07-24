# JSON Schema 索引

| 文件 | 责任方 | 用途 |
|---|---|---|
| `agent-config.schema.json` | A | 总 Agent 配置 |
| `task.schema.json` | A | 用户任务输入 |
| `task-plan.schema.json` | A | 语义 TaskPlan |
| `online-observation.schema.json` | B/C/F | 在线传感观测；禁止 GT |
| `executor-health.schema.json` | D/E | VLA 服务身份与健康 |
| `executor-infer.schema.json` | A/D/E | 推理请求/响应信封 |
| `executor-cancel.schema.json` | A/D/E | 幂等取消 |
| `action-chunk.schema.json` | A/B/D/E | 统一 N×7 动作 |
| `verify.schema.json` | A/F | 在线后置条件核验 |
| `event.schema.json` | A/F | 结构化事件证据 |

CI 会用 Draft 2020-12 元 Schema 校验所有文件，并验证默认 Agent 配置。
Schema 变化必须先更新接口文档和契约测试，再修改生产服务。

所有 `checkpoint_sha` 与 `norm_stats_sha` 必须使用完整
`sha256:<64 位十六进制>`；配置文件仅允许明确的待替换占位符，任何真实服务
启动时都会拒绝该占位符以及 `latest`、版本昵称和缩写摘要。
