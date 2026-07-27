# Simulation

负责人：B。本目录保存仿真环境、控制器、场景配置和总 Agent 的环境适配，不保存
生成缓存、录包或大体积导出资产。

推荐结构：

```text
simulation/
├── README.md
├── adapters/       # observe/step/safe-stop 接口实现
├── configs/        # Isaac Sim 的冻结默认/示例配置
├── controllers/    # Franka/夹爪控制与限幅
├── scenes/         # 可版本化的场景描述和资产清单
└── tests/          # headless、重启、坐标系和安全回归
```

G0 后只能保留一个主仿真平台；不要同时维护重复的 Isaac/Gazebo 生产路径。每次正式
实验必须记录平台版本、物理参数、控制频率、坐标系、随机 seed 和资产 SHA。
`cache/`、`generated/`、`packages/`、录像及机器人录包均由 `.gitignore` 排除。

## 冻结 MVP 场景：双 Franka + 单箱固定交接

当前代码只实现已经冻结的场景，不再引入传送带、多料箱或第三台机械臂：

- Arm A（π0.5）完成 4 个零件装箱，并把满箱放到中央固定交接位；
- Supervisor 完成 `A_ONLY → HANDOFF_VERIFY → B_ONLY` 令牌切换；
- Arm B（OpenVLA）在收到 `handoff_ready` 后把同一料箱搬到成品位；
- 三台固定相机同时服务 VLA 观测、YOLO 检测证据和闭环验证。

本目录的第一版场景工具如下：

```text
simulation/
├── configs/single_bin_scene_v1.json  # 唯一场景坐标合同
├── scene_layout.py                   # 无 Isaac Sim 也可运行的静态预检
├── isaac_compat.py                   # Isaac Sim 4.2/4.5/5.1 薄兼容层
├── single_bin_scene_builder.py       # USD 几何、物理、相机与机器人构建
└── build_single_bin_scene.py         # Standalone Python 入口
```

本轮交付范围是“可导入的场景 USD + Camera Prim”。它还没有创建运行期
RenderProduct、图像缓冲区或 YOLO 订阅；这些应在场景、IK 和遮挡检查通过后再接，
避免把视觉服务问题与基础场景问题混在一起排查。

成员 B 补做 D00-D03 时，使用
[`run_g0_acceptance.py`](run_g0_acceptance.py) 完成 1000 步、20 次 Reset、
双臂状态和三相机样本的 G0 验收。Linux 一键入口与逐步说明分别是：

```bash
ISAAC_SIM_ROOT=/absolute/path/to/isaacsim \
  bash scripts/run_g0_linux.sh
```

- [`../scripts/run_g0_linux.sh`](../scripts/run_g0_linux.sh)
- [`../docs/simulation/member-b-catch-up-guide.md`](../docs/simulation/member-b-catch-up-guide.md)

原始结果写入被 Git 忽略的 `artifacts/g0/`；只提交填写后的 Markdown、代码和
小型证据索引，不能把失败结果手工改成通过。

主开发与最终 Docker 建议冻结 **Isaac Sim 5.1.x**。代码兼容 4.5，并为 4.2
保留最低限度的导入回退；不要把多个 Isaac Sim 版本混入同一个正式镜像。

### 1. 先做本地合同与距离预检

这一步只需要普通 Python 3.10+：

```powershell
python simulation\scene_layout.py `
  --config simulation\configs\single_bin_scene_v1.json

python -m pytest tests\test_scene_layout.py
```

通过仅表示场景数量、坐标、区域配方和水平软半径没有明显错误，不代表机器人 IK、
碰撞或抓取物理已经通过。

### 2. 在 Isaac Sim 中生成 USD

若使用 NVIDIA 安装包，在 PowerShell 中将路径替换为自己的 Isaac Sim 安装目录：

```powershell
$IsaacSimRoot = "C:\isaacsim"
& "$IsaacSimRoot\python.bat" simulation\build_single_bin_scene.py `
  --config simulation\configs\single_bin_scene_v1.json `
  --output simulation\generated\single_bin_scene_v1.usda
```

若使用 Isaac Sim Python 环境，激活环境后直接运行：

```powershell
python simulation\build_single_bin_scene.py `
  --config simulation\configs\single_bin_scene_v1.json `
  --output simulation\generated\single_bin_scene_v1.usda
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--headless` | 无界面生成场景，适合 CI/Docker |
| `--franka-usd <URI或路径>` | 官方 Franka 资产不在默认位置时显式指定 |
| `--no-robots` | 只生成工作台、物体和相机，用于先检查几何布局 |

默认会依次探测新版
`/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` 和旧版
`/Isaac/Robots/Franka/franka.usd`。找不到资产时脚本会明确失败，不会悄悄生成两台
“空机器人”。

### 3. 打开并复核

在 Isaac Sim 中选择 **File → Open**，打开：

```text
simulation/generated/single_bin_scene_v1.usda
```

重点检查以下 Prim：

```text
/World/Robots/Arm_A
/World/Robots/Arm_B
/World/Parts/P01 ... P04
/World/Bins/Bin_01
/World/Stations/PACK_STATION
/World/Stations/HANDOFF_CENTER
/World/Stations/FINISHED_01
/World/Cameras/CAM_A_TOP
/World/Cameras/CAM_HANDOFF
/World/Cameras/CAM_B_TOP
```

料箱由底板、四壁、隔板和前侧把手组成，共用一个刚体根节点；没有使用封闭整个
开口的单一凸包，因此零件可以真正落入 2×3 格口。

三个 Station Prim 的 Xform 保留的是“料箱中心目标位姿”（`z=0.785 m`）；桌面
色块是各 Station 下的 `Marker` 子节点。后续控制器应读取 Station 根节点，不能把
色块表面高度误当成料箱中心高度。

### 4. Isaac Sim 内的硬验收

生成 USD 后，必须按顺序完成：

1. A 臂对 P01～P04 的预抓取、抓取、抬升和箱内放置位逐点 IK；
2. A 臂把满箱搬到 `HANDOFF_CENTER`，然后完全退出共享区；
3. B 臂只在 `handoff_ready` 后抓取前侧把手并搬到 `FINISHED_01`；
4. 完整机械臂碰撞体不能同时进入共享区；
5. 连续 Reset 20 次无穿模、弹飞和 NaN；
6. 教师策略完整闭环至少连续 3 次成功，再接入双 VLA。

最终 Docker 不应依赖比赛现场联网下载 Franka 资产。发布前应在断网环境重新打开
生成的 USD，并检查两台 Articulation、外部引用和三台相机都完整。
