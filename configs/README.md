# 配置说明

`agent.default.json` 是唯一正式运行配置，固定为 V2
`single_bin_manual_industrial_v2`：

- 总控只装配 `pi05`；P01/W01 发放 `A_ONLY`，任务三按交接阶段发放 `A_ONLY`/`B_ONLY`；
- 正式任务为 `P01_TO_S11`、`W01_TO_S14` 和 `BIN01_TO_FINISHED01`；
- 每次只执行一个 7D 微动作，然后重新观测；
- 终局证据固定使用 3 帧、至少 2 票、置信度不低于 0.6；
- 决策预算耗尽、观测合同错误或安全异常时必须确认 safe-stop；
- 在线观测使用 `schemas/online-observation-v2.schema.json`，禁止 V1 生命周期字段和 GT。

部署前必须把 `executors.pi05.checkpoint_sha` 与 `norm_stats_sha` 替换为完整的
`sha256:<64 位十六进制>` 固定摘要。任何 `latest`、版本昵称、缩写摘要或占位符
都会被 Schema 或适配器拒绝。

`agent.v2.default.json` 是便于显式引用的同内容副本；修改时必须保持与
`agent.default.json` 一致。`v2-task-profile.json` 是任务 ID、用户指令、对象和槽位
的一一对应真源。五条 UI 指令中，尚未具备正式数据合同的两条不得进入推理。

仓库不再保留旧版 Agent 配置。生产入口只接受 V2 配置，模型服务的 YOLO 参数
单独位于 `perception.yolo-*.json`，π0.5 参数位于 `agent.default.json`。
