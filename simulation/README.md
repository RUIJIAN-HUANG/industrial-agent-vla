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

## 当前场景：V2 人工工业采集

> 实现状态（2026-08-19）：下列 V2 场景配置、构建器和 GUI 验收/采集脚本是 B 侧
> 计划交付物，当前仓库尚不存在。现有 `v2_collection_recorder.py` 只提供经过测试的
> Canonical V2 同步写入边界，不构成可运行场景或正式采集入口。

当前主场景 ID 为 `single_bin_manual_industrial_v2`。它保留双 Franka、三相机、
单箱固定交接的安全框架，并把采集对象升级为 8 个工业零件和 `2×4` 料箱：

- P01/P02 为正立轴件，P03/P04 为倒立轴件；
- N01/N02 为带可见通孔的六角螺母；
- W01/W02 为带平行手柄和开口端的简化扳手；
- A/B/C/D 四区各 2 件，S11-S24 与 8 件一一固定映射；
- 料箱带中央提梁和 `BIN_CARRY_TCP`，计划满载质量 `1.0 kg`；
- 键盘采集以 `10 Hz` 写入 Canonical Episode，在线 GT 被禁止。

V2 计划文件（除 `v2_collection_recorder.py` 外尚待 B 提交）：

```text
simulation/
├── configs/single_bin_scene_v2.json       # V2 场景机器真源
├── v2_scene_contract.py                   # 数量、坐标、槽位、质量、相机合同
├── v2_industrial_assets.py                # 轴件、螺母、扳手程序化资产
├── single_bin_scene_v2_builder.py         # 2×4 料箱、提梁、机器人、相机构建
├── build_single_bin_scene_v2.py           # V2 USD 构建入口
├── run_v2_gui_scene_acceptance.py         # 可见 GUI + 四张图证据
├── run_v2_home_acceptance.py              # 两臂 HOME
├── run_v2_ik_reachability_acceptance.py   # IK 可达性
├── run_v2_dual_arm_micro_motion_acceptance.py
├── run_v2_keyboard_collection.py          # 人工键盘 Canonical Episode
├── v2_collection_preflight.py             # 正式采集预检
├── v2_collection_state.py                 # V2 人工采集状态机
└── v2_collection_recorder.py              # 已实现：Canonical V2 同步写入边界
```

以下静态预检命令须待对应 B 侧脚本提交后使用：

```powershell
python simulation\run_v2_scene_acceptance.py `
  --evidence-dir artifacts\v2\static
```

使用 Isaac Sim Python 运行可见 GUI 验收：

```powershell
python simulation\run_v2_gui_scene_acceptance.py `
  --output-scene simulation\generated\single_bin_scene_v2.usda `
  --evidence-dir artifacts\v2\gui `
  --review-seconds 45
```

V2 的静态 PASS 不代表 GUI、物理、IK、抓取或满载搬运通过。正式顺序与状态声明
见 [`../docs/v2-manual-industrial-collection.md`](../docs/v2-manual-industrial-collection.md)。

## V1 自动闭环兼容基线

`single_bin_pack_handoff_v1` 仍用于四 Agent 自动串行闭环：Arm_A/π0.5 处理
P01-P04，Supervisor 执行三帧交接核验，Arm_B/OpenVLA-OFT 搬运同一料箱。
V1 的 `2×3` 料箱、冻结指令和后置条件未被 V2 静默替换。

V1 场景工具如下：

```text
simulation/
├── configs/single_bin_scene_v1.json  # V1 自动闭环场景坐标合同
├── scene_layout.py                   # 无 Isaac Sim 也可运行的静态预检
├── isaac_compat.py                   # Isaac Sim 4.2/4.5/5.1 薄兼容层
├── single_bin_scene_builder.py       # USD 几何、物理、相机与机器人构建
├── rgb_cas_bridge.py                 # RGB/RGBA Annotator → 共享图像 CAS
├── isaac_franka_controller.py        # Isaac 5.1 Lula/Franka owner-thread 后端
├── run_isaac_adapter_smoke.py        # 双臂分别执行的 Gate/微动作/急停 smoke
└── build_single_bin_scene.py         # Standalone Python 入口
```

执行 Adapter 的线程 Gate、durable command journal、Linux 双臂 smoke 和明确未覆盖
范围见 [`../docs/simulation/isaac-execution-adapter.md`](../docs/simulation/isaac-execution-adapter.md)。

场景构建入口负责“可导入的场景 USD + Camera Prim”，尚不自动创建运行期
RenderProduct。创建 RGB Annotator 后，必须把其 `uint8 H×W×3/4` 输出传给
`IsaacRgbCasPublisher.publish()`；该桥会严格校验冻结相机分辨率、移除 alpha、
编码 RGB PNG、原子写入共享 CAS 并返回 `ImageReference`。不得在仿真适配器中
另写一套路径或 SHA 逻辑。

场景 JSON 只保存几何、物理、相机和场景事件名。交接稳定帧数属于 Supervisor
生命周期配置，机械臂正常工作空间属于控制器安全配置，两者的机器真源均为
`configs/agent.default.json`；不得在场景 JSON 中重复维护
`handoff_verify_stable_cycles` 或 `normal_workspace_limits`。
Reset 后的物理稳定步数也不属于静态 USD 场景合同；应由 G0 运行/验收入口通过
显式参数执行并记录。在该运行入口落地前，不得用场景 JSON 中未消费的
`reset_settle_steps` 宣称已经完成自动稳定。

成员 B 补做 D00-D03 时，使用
[`run_g0_acceptance.py`](run_g0_acceptance.py) 完成 1000 步、20 次 Reset、
双臂状态和三相机样本的 G0 验收。Linux 一键入口与逐步说明分别是：

```bash
EXPECTED_GIT_SHA="$(git rev-parse HEAD)" \
  ISAAC_SIM_ROOT=/absolute/path/to/isaacsim \
  bash scripts/run_g0_linux.sh
```

- [`../scripts/run_g0_linux.sh`](../scripts/run_g0_linux.sh)
- [`../docs/simulation/member-b-catch-up-guide.md`](../docs/simulation/member-b-catch-up-guide.md)

原始结果写入被 Git 忽略的 `artifacts/g0/`；只提交填写后的 Markdown、代码和
小型证据索引，不能把失败结果手工改成通过。

主开发与最终 Docker 建议冻结 **Isaac Sim 5.1.x**。代码兼容 4.5，并为 4.2
保留最低限度的导入回退；不要把多个 Isaac Sim 版本混入同一个正式镜像。

### V1-1. 先做本地合同与距离预检

这一步只需要普通 Python 3.10+：

```powershell
python simulation\scene_layout.py `
  --config simulation\configs\single_bin_scene_v1.json

python -m pytest tests\test_scene_layout.py
```

通过仅表示场景数量、坐标、区域配方和水平软半径没有明显错误，不代表机器人 IK、
碰撞或抓取物理已经通过。

### V1-2. 在 Isaac Sim 中生成 USD

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

### V1-3. 打开并复核

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

### V1-4. Isaac Sim 内的硬验收

生成 USD 后，必须按顺序完成：

1. A 臂对 P01～P04 的预抓取、抓取、抬升和箱内放置位逐点 IK；
2. A 臂把满箱搬到 `HANDOFF_CENTER`，然后完全退出共享区；
3. B 臂只在 durable `handoff.ready` 事件确认后抓取前侧把手并搬到 `FINISHED_01`；
4. 完整机械臂碰撞体不能同时进入共享区；
5. 连续 Reset 20 次无穿模、弹飞和 NaN；
6. 教师策略完整闭环至少连续 3 次成功，再接入双 VLA。

最终 Docker 不应依赖比赛现场联网下载 Franka 资产。发布前应在断网环境重新打开
生成的 USD，并检查两台 Articulation、外部引用和三台相机都完整。
