# Configurations

`agent.default.json` 定义总 Agent 的恢复、安全和双执行器默认值。部署前必须：

1. 将两个 `checkpoint_sha` 和 `norm_stats_sha` 占位符替换为
   `sha256:<64 位十六进制>` 的完整固定摘要；
2. 由 B/D/E 共同冻结坐标系、单位、轴限幅和夹爪正负语义；
3. 为每次正式实验保存不可变配置副本并记录 Git commit；
4. 不在配置中提交 token、密码或机器私有路径。

修改动作合同主版本、恢复预算或工作空间属于接口变更，必须同步更新
`schemas/`、接口文档和契约测试。

真实部署先用 `build_executors_from_config(config, transport_factory)` 消费每个
`base_url` 并构建独立进程适配器，再用
`IndustrialAgent.from_config(executors, config)` 加载总 Agent。后者会逐执行器
比对名称、动作合同、checkpoint SHA 和 norm stats SHA；不一致时启动失败。

代码还会拒绝打开回切、增加切换次数或关闭恢复清队列等违反冻结不变量的配置。
任何 `latest`、版本昵称、缩写摘要或占位符都会被 Schema、适配器构造器和
Agent 启动校验拒绝。
