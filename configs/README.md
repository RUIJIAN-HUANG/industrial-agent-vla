# 配置说明

`agent.default.json` 定义四 Agent、双 Franka 固定串行生命周期，以及恢复、
安全、YOLO 评分 sidecar 和两个固定 VLA 执行器。部署前必须：

1. 将两个 VLA 的 `checkpoint_sha`、`norm_stats_sha`，以及 YOLO 的
   `checkpoint_sha`、`class_map_sha`、`config_sha` 占位符替换为
   `sha256:<64 位十六进制>` 的完整固定摘要；
2. 由 B/D/E 共同冻结坐标系、单位、轴限幅和夹爪正负语义；
3. 由 A/B/F 冻结令牌顺序、交接票数和
   `robot.arm_a.retreated`、`robot.arm_b.retreated` 的布尔语义；
4. 为每次正式实验保存不可变配置副本并记录 Git commit；
5. 不在配置中提交 token、密码或机器私有路径。

## 冻结不变量

- `lifecycle.mode` 必须为 `FIXED_DUAL_VLA_SERIAL`；
- `pi05` 必须启用并且只控制 Arm_A；
- `openvla_oft` 必须启用并且只控制 Arm_B；
- 令牌顺序固定为 `A_ONLY → HANDOFF_VERIFY → B_ONLY → NONE`；
- `max_switches_per_run` 必须为 `0`；
- 两个 VLA 都是同一闭环的必要固定阶段，不提供单模型运行模式；
- 当前子任务失败时只能在本阶段预算内重新观察并重试，耗尽后安全停止，
  不允许另一个 VLA 接管。

`perception.required: true` 表示四 Agent 拓扑在启动时必须注入 YOLO Agent
适配器，不表示某次检测成功是动作或令牌的硬依赖。当前实现对新鲜帧同步调用
YOLO sidecar；空检测、超时或坏包只记录评分证据失败，π0.5、OpenVLA-OFT
和固定令牌生命周期继续按安全条件执行。

修改动作合同主版本、恢复预算、工作空间、交接票数或 YOLO 类别表属于接口
变更，必须同步更新 `schemas/`、接口文档和契约测试。

真实部署先用 `build_executors_from_config(config, transport_factory)` 消费两个
VLA 的 `base_url` 并构建独立进程适配器，再用
`IndustrialAgent.from_config(...)` 加载 Supervisor。启动过程会逐执行器比对
名称、固定职责、动作合同、checkpoint SHA 和 norm stats SHA；传入执行器集合
必须与配置中启用的两个固定名称完全一致。

YOLO 使用独立适配器消费 `perception.base_url`。启动时校验 Agent 名称、
检测合同和三个身份 SHA；运行时保存同帧关联信息与原始预测，但不把 bbox
作为 VLA 必需输入。

任何 `latest`、版本昵称、缩写摘要或占位符都会被 Schema、适配器构造器或
Agent 启动校验拒绝。
