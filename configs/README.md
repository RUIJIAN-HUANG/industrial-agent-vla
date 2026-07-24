# Configurations

`agent.default.json` 定义总 Agent 的恢复、安全和双执行器默认值。部署前必须：

1. 将两个 `checkpoint_sha` 和 `norm_stats_sha` 占位符替换为真实、固定摘要；
2. 由 B/D/E 共同冻结坐标系、单位、轴限幅和夹爪正负语义；
3. 为每次正式实验保存不可变配置副本并记录 Git commit；
4. 不在配置中提交 token、密码或机器私有路径。

修改动作合同主版本、恢复预算或工作空间属于接口变更，必须同步更新
`schemas/`、接口文档和契约测试。

核心参数通过 `IndustrialAgent.from_config(executors, config)` 加载。代码会拒绝
打开回切、增加切换次数或关闭恢复清队列等违反冻结不变量的配置。`executors`
中的 URL/SHA 由 D/E 的服务启动器消费，不能用占位符运行真实实验。
