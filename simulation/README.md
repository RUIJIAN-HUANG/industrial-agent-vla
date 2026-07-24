# Simulation

负责人：B。本目录保存仿真环境、控制器、场景配置和总 Agent 的环境适配，不保存
生成缓存、录包或大体积导出资产。

推荐结构：

```text
simulation/
├── README.md
├── adapters/       # observe/step/safe-stop 接口实现
├── configs/        # Isaac 或 Gazebo 的冻结默认/示例配置
├── controllers/    # Franka/夹爪控制与限幅
├── scenes/         # 可版本化的场景描述和资产清单
└── tests/          # headless、重启、坐标系和安全回归
```

G0 后只能保留一个主仿真平台；不要同时维护重复的 Isaac/Gazebo 生产路径。每次正式
实验必须记录平台版本、物理参数、控制频率、坐标系、随机 seed 和资产 SHA。
`cache/`、`generated/`、`packages/`、录像及机器人录包均由 `.gitignore` 排除。
