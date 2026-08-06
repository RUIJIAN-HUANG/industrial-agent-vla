# 成员 B：Isaac Sim 数据接口交付与阻塞清单（2026-08-01）

> 用途：在 Linux Isaac Sim 主机暂不可用期间，向 A/C/D/E/F 提供成员 B
> 已经能够冻结的场景、相机、动作和执行器事实，并明确哪些结论仍需 Linux
> 实机验证。本文件不是正式数据 Gate，也不表示已经具备批量采集条件。
>
> 最后远端复核：2026-08-02；已联网执行 `git fetch --all --prune`，代码基线为
> `origin/main@e00675a`。

## 1. 一页结论

- 场景静态配置、三台顶视/交接 RGB 相机、双 Franka 结构和 7 维动作顺序已经有
  仓库机器真源。
- 执行适配器 PR #12 已合入 `main`；合并提交为 `3b68e5e`，适配器最终提交为
  `3c9ce15`。2026-08-02 联网 fetch 后，本地 `main` 与 `origin/main` 一致。
- Windows 可执行的场景静态校验通过；执行适配器、控制器、主线程 Gate、G0 和
  Isaac 兼容层共 70 项单元测试通过。
- 当前代码尚无自建场景键盘入口、Canonical Recorder 和真实 Episode。
- 当前适配器 smoke 已输出关节位置/速度和 robot-base TCP 位姿，但仍是
  controller-only smoke，`camera={}`，没有三路 RGB 证据。因此 C/D/E/F 可以先用
  synthetic Canonical Episode 开发和测试，但真实数据 Gate 必须等待 Linux 主机恢复。
- 当前仍有三个需要由 A 组织冻结的 P0 接口问题：腕相机口径、Canonical TCP 到
  模型 state 的转换、动作块长度/采样频率。夹爪硬件边界映射已在 `3c9ce15` 冻结。

## 2. 代码基线与场景版本

| 项目 | 当前值 | 证据 |
|---|---|---|
| 仓库 | `RUIJIAN-HUANG/industrial-agent-vla` | Git remote 对应团队仓库 |
| 当前基线分支 | `main` | 2026-08-02 已从旧功能分支切回 |
| 当前 HEAD / `origin/main` | `e00675a1da3352c07500f2dedb3a5ed2a7463661` | 联网 `git fetch --all --prune` 后核对 |
| 执行适配器最终提交 | `3c9ce15ef01b69da49300c9addb4a226c2986c0a` | 已被 `origin/main` 包含 |
| PR #12 合并提交 | `3b68e5e8cb626cd34a365a0a309f2a713f9c9c55` | `Merge pull request #12` |
| 当前分支同步状态 | `main` 与 `origin/main` 为 `0 ahead / 0 behind` | 2026-08-02 联网检查 |
| 场景配置 | `simulation/configs/single_bin_scene_v1.json` | 仓库机器真源 |
| 场景 ID | `single_bin_static_handoff_v1` | 场景 JSON 的 `scene_id` |
| Schema 版本 | `1.0` | 场景 JSON 的 `schema_version` |
| 场景配置 SHA-256 | `16e7060ef3fd0997ae2ee958983a8d45926055c9989a11967a783d641d0d0119` | 2026-08-01 `Get-FileHash -Algorithm SHA256` |

使用方应基于 `origin/main` 的完整 HEAD，而不是继续依赖已经合并的旧功能分支或
复制成员 B 当前工作目录。当前本地工作区存在未跟踪文档，不应把本地工作区状态
当作可复现基线。

## 3. 相机合同

| 相机 ID | 分辨率 | 用途 | 当前状态 |
|---|---:|---|---|
| `CAM_A_TOP` | 1280×720 RGB | Arm_A / π0.5 主视角 | 场景配置已定义；G0 曾取证 |
| `CAM_HANDOFF` | 1280×720 RGB | 固定交接区证据 | 场景配置已定义；G0 曾取证 |
| `CAM_B_TOP` | 1280×720 RGB | Arm_B / OpenVLA 主视角 | 场景配置已定义；G0 曾取证 |

冻结 MVP 当前没有以下 Camera Prim：

```text
CAM_A_WRIST
CAM_B_WRIST
```

因此当前统一模型接口必须保留 `wrist_image` 字段，但值应为 JSON `null`。D 的
2026-08-01 输入说明中“训练时固定使用 `CAM_B_WRIST`”与当前场景、架构文档、
OpenVLA 配置和测试冲突。在 A 批准架构变更之前，不得据此增加腕部相机或开始
采集腕部图像。

场景配置还声明：

```text
physics_dt_s         = 1/120 s
rendering_dt_s       = 1/30 s
control_frequency_hz = 60 Hz
```

这些是配置目标，不等于真实 Recorder 已按该频率稳定输出；实际时间戳和掉帧仍需
Linux 实测。

## 4. 统一 7 维动作合同

动作空间：`ee_delta_pose_gripper`。

```text
索引 0: dx_m
索引 1: dy_m
索引 2: dz_m
索引 3: dax_rad
索引 4: day_rad
索引 5: daz_rad
索引 6: gripper_norm
```

其中 `[dax_rad, day_rad, daz_rad]` 三个数共同组成当前机械臂 `robot_base`
坐标系下的 rotation-vector（轴角旋转向量）增量：向量方向表示旋转轴，向量模长
表示旋转角度。它们不是三次独立的 `droll/dpitch/dyaw` 欧拉角旋转，采集端、
Canonical、RLDS 和 LeRobot 均不得自行改回欧拉角动作语义。

这里必须区分 **action** 与 **state**：动作合同已经冻结为 rotation-vector；模型
本体状态是否使用 `roll/pitch/yaw` 是另一个字段问题，仍需 C/D/E 按现有模型配置
完成明确的转换与往返测试。

机器合同：

```text
frame            = robot_base
translation_unit = m
rotation_unit    = rad
gripper_unit     = normalized
duration_ms      = 1..10000
动作块长度       = 1..32（Schema 上限；训练时固定 N 尚未冻结）
```

机器 Schema 当前规定 `gripper_norm ∈ [-1, 1]`。

## 5. 当前控制器的夹爪实际行为

`simulation/isaac_franka_controller.py` 在提交 `3c9ce15` 中已经冻结二值硬件
边界，当前执行逻辑是：

```python
finger_position_m = 0.04 if float(command) >= 0.5 else 0.0
```

因此实际语义为：

| 输入 | 当前执行结果 |
|---:|---|
| `< 0.5` | 每个手指目标位置 `0 m`，闭合 |
| `>= 0.5` | 每个手指目标位置 `0.04 m`，打开 |

该规则有专门单元测试，统一映射 π0.5 的 `0/1` 与 OpenVLA-OFT 的 `-1/+1`
端点。夹爪命令在硬件边界是**二值**而非连续位置：`-1`、`0` 均闭合，`1` 打开。
转换器必须保留这一语义，不能把中间负值解释为连续手指宽度。

## 6. 已通过的检查

检查日期：2026-08-01。执行环境：Windows；不加载真实 Isaac Sim runtime。

### 6.1 场景静态校验

命令：

```powershell
python simulation/scene_layout.py `
  --config simulation/configs/single_bin_scene_v1.json `
  --json
```

结果：

```text
valid=true
errors=[]
Arm_A/Arm_B 所列目标均在 0.65 m 软半径内
```

静态校验只证明配置结构和水平可达距离没有明显错误，不证明 IK、碰撞、抓取或
Recorder 已通过。

### 6.2 单元测试

测试模块：

```text
tests.test_isaac_environment
tests.test_isaac_franka_controller
tests.test_isaac_runtime
tests.test_g0_acceptance
tests.test_isaac_compat
```

结果：

```text
Ran 70 tests
OK
```

覆盖内容包括：

- observation/command ID 与 digest 防陈旧检查；
- 控制令牌、双臂退避互锁和失败安全停止；
- 重复命令拒绝；
- base-frame 增量到 world-frame 的数学转换；
- robot-base TCP 位姿与旋转向量生成；
- π0.5 `0/1` 与 OpenVLA `-1/+1` 的二值夹爪映射；
- Isaac 主线程 Gate、超时、取消和并发安全；
- safe-stop hold target 回读；
- G0 相机分辨率、非纯色帧和像素摘要检查；
- Isaac Sim 5.1 Stage/API 兼容路径。

以上均为单元或静态测试，不能替代当前 HEAD 的 Linux Isaac Sim 5.1 实机 smoke。

## 7. 当前代码能提供与不能提供的真实数据

| 数据/能力 | 当前代码状态 | 对下游的影响 |
|---|---|---|
| 双 Franka 场景和三相机 Prim | 已有 | C 可据此固定 ID 和目录 |
| 关节位置/速度读取 | adapter smoke 已有 | 可先定义 Canonical 字段 |
| 7D 动作执行 | 代码和单测已有 | 可制作 synthetic action fixture |
| robot-base TCP 位姿进入 smoke observation | 已有代码和单测 | 可定义 state；真实数值待 Linux 验证 |
| RGB RenderProduct/Annotator 接入 adapter observation | **尚无** | D/E 所需真实图像样本无法生成 |
| 自建场景键盘入口 | **尚无** | 不能在自建场景完成键采 |
| Canonical Recorder | **尚无；C 主责** | 无正式 Episode |
| Episode Replay | **尚无；C 主责** | 无法验证动作/状态一致性 |
| Canonical→RLDS Loader 闭环 | **待 D** | Arm_B 训练链未放行 |
| Canonical→LeRobot Loader 闭环 | **待 E** | Arm_A 训练链未放行 |

## 8. 仍需 Linux Isaac Sim 5.1 实测的内容

1. 当前主线（adapter `3c9ce15`，基线 `e00675a`）的 GUI supervised adapter smoke。
2. 当前主线的 headless adapter smoke。
3. `Arm_A`、`Arm_B` 小幅增量动作是否产生可测关节变化。
4. Lula IK 在冻结场景两台偏置基座上的真实收敛情况。
5. 夹爪目标是否被 articulation 正确执行和回读。
6. safe-stop hold target 是否在真实 Isaac Sim 5.1 中确认。
7. 自建场景键盘移动、旋转、夹爪、Reset 和退出。
8. 三路 RGB 的真实时间戳、掉帧、重复帧和黑帧情况。
9. TCP pose、夹爪、关节、action 与 RGB 的同步记录。
10. 一条真实 Canonical Episode 的完整性、回放和跨格式转换。
11. 20 回合脚本教师报告与 50 局 G1 Oracle Gate。

## 9. 字段对齐表 v0.1

状态含义：`确认`表示已有机器真源；`待负责人确认`表示不得据此批采；`待 Linux`
表示接口可先开发但真实值未验证。

| 数据含义 | B / Isaac 输出 | C / Canonical | D / OpenVLA | E / LeRobot | 状态与负责人 |
|---|---|---|---|---|---|
| Episode ID | 运行时生成 | `meta.episode_id` | `task_id` 前缀或样本关联，待 D 明确 | 待 E 明确 | C/D/E |
| 场景版本 | config SHA-256、Git SHA | `scene_config_sha256`、`controller_version` | 应进入 manifest | 应进入 manifest | B 值已确认；C/D/E 映射待确认 |
| Arm_B 顶视图 | `CAM_B_TOP`, 1280×720 RGB | `rgb.CAM_B_TOP` 相对路径 | `full_image` | 不属于 Arm_A 主数据 | ID/尺寸确认；编码/CAS 映射待 C/D |
| Arm_A 顶视图 | `CAM_A_TOP`, 1280×720 RGB | `rgb.CAM_A_TOP` 相对路径 | 不属于 Arm_B 主数据 | full image 字段待 E | ID/尺寸确认；E 映射待确认 |
| 交接图 | `CAM_HANDOFF`, 1280×720 RGB | `rgb.CAM_HANDOFF` | 是否进入训练待 D；在线不能替代 B 顶视图 | 是否进入训练待 E | C/D/E |
| 腕图 | 当前无 Camera Prim | 不创建目录或明确 null | 当前应 `wrist_image=null` | 当前应 `wrist_image=null` | D 文档冲突；A 决策，D/E 修订 |
| 关节位置 | live articulation | `joint_position` | 当前模型 state 不直接采用，待 D | 待 E | B/C/D/E |
| 关节速度 | live articulation | `joint_velocity` | 是否使用待 D | 待 E | B/C/D/E |
| TCP 位姿 | smoke 输出 `tcp_pose_m_rad=[x,y,z,rx,ry,rz]`，robot base | 现文档为 `[x,y,z,qx,qy,qz,qw]` | D 文档为 `[x,y,z,roll,pitch,yaw,gripper]` | 待 E | 真实观测已有；A/C/D/E 冻结跨表示转换 |
| 夹爪状态 | `gripper_open` 布尔；命令 `<0.5` 闭合、`>=0.5` 打开 | `gripper_state` | 7D state 最后一维 | 待 E | 硬件边界已确认；C/D/E 对齐存储表示 |
| 单步动作 | `[dx,dy,dz,dax,day,daz,gripper]`；后三维为 robot-base rotation-vector | `action_7d`，必须保持同一语义 | 聚合为 `N×7`，不得当作欧拉角动作 | 待 E 按同一动作合同确认 | 顺序/旋转语义已冻结；N/FPS 待 A/C/D/E |
| 动作时长 | `duration_ms` | `action_duration_s` | chunk 时间语义待 D | FPS/时间语义待 E | A/C/D/E |
| 指令 | Supervisor/TaskProfile 提供 | `meta.instruction` | `task_description` | 字段名待 E | A/C/D/E |
| Seed | scene/reset 输入 | `meta.scene_seed`，采前分配 | 保留可追溯性 | 保留可追溯性 | C/F 主责；B 执行 |
| Split | B 不决定 | 采集前写 `train/val/test` | 不得重新随机拆帧 | 不得重新随机拆帧 | C/F |
| 终局 | 仿真/任务判定信号待接入 | `outcome`、failure label | 训练过滤规则待 D | 训练过滤规则待 E | A/C/F/D/E |
| 在线 GT | B 不进入 observation | 只能在隔离 `offline_gt/` | 禁止输入 | 禁止输入 | C/F 验收 |

## 10. 阻塞问题清单

| ID | 优先级 | 阻塞问题 | Owner / 决策人 | 放行证据 | 未解决时的处理 |
|---|---|---|---|---|---|
| DATA-BLOCK-01 | P0 | D 文档要求 `CAM_B_WRIST`，冻结场景无腕相机 | A 决策；D 修订；B/C配合 | 架构、配置、Schema、测试一致 | 保持 `wrist_image=null`，禁止加腕相机 |
| DATA-BLOCK-03 | P0 | Canonical TCP 四元数与 D 的欧拉角 state 缺少冻结转换 | A/C/D；B提供真实值 | 坐标/单位/旋转顺序测试，往返误差报告 | 只用 synthetic 数据开发，不训练 |
| DATA-BLOCK-04 | P0 | 动作块 N、采样频率、降采样和末尾 padding 未冻结 | A/C/D/E；B提供运行频率 | 同一 Episode 跨格式 step 对齐测试 | 禁止批采 |
| DATA-BLOCK-05 | P0 | 没有 Canonical Recorder 和 Replay | C；B提供仿真接口；F验收 | 1 条 synthetic + 1 条真实 Episode 可回放 | 禁止正式 Episode |
| DATA-BLOCK-06 | P0 | adapter smoke 已有 TCP，但 observation 仍无三路 RGB/CAS | B；C评审 Recorder 接口 | 真实 RGB observation Schema + 单测 + Linux 样例 | D/E仅使用 synthetic fixture |
| DATA-BLOCK-07 | P0 | 自建场景没有键盘入口 | B | GUI下移动/旋转/夹爪/Reset/安全退出日志 | 不开展自建场景键采 |
| DATA-BLOCK-08 | P0 | 当前 adapter HEAD 未做 Linux 实机 smoke | B；F复核 | GUI + headless PASS JSON 和日志 | 不宣称执行适配器实机完成 |
| DATA-BLOCK-09 | P0 | D/E Loader 尚未以同一 Canonical 样例通过 | D/E；C提供 fixture；F验收 | RLDS/LeRobot Loader 输出与往返测试 | 禁止批量采集 |
| DATA-BLOCK-11 | P1 | Linux 主机暂不可用 | A协调资源；B恢复后执行 | 可用时间窗口和实机运行记录 | 先完成 synthetic Gate |

已经解除的旧阻塞：

- `DATA-BLOCK-02`：`3c9ce15` 已冻结二值夹爪硬件映射并补单元测试。
- `DATA-BLOCK-10`：PR #12 已通过 `3b68e5e` 合入 `main`。

## 11. 不依赖 Linux、可以立即并行的交付

| 成员 | 现在可完成 | 需要 B 提供什么 |
|---|---|---|
| A | 冻结腕相机、state、N/FPS 三项决策 | 本文件和对应代码路径；夹爪硬件映射已冻结 |
| C | Schema、Recorder 接口、synthetic Canonical Episode、Replay/完整性测试骨架 | 相机 ID、动作顺序、场景/控制器 SHA；真实值稍后补 |
| D | 修订 OpenVLA 输入文档；用 synthetic Episode 完成 Canonical→RLDS/Loader | Arm_B 字段语义和本文件；不需要等待真实 Linux 数据 |
| E | 提供 LeRobot features；用 synthetic Episode完成转换/Loader/norm stats smoke | Arm_A 字段语义和本文件；不需要等待真实 Linux 数据 |
| F | 编写黑帧、重复帧、时间戳、NaN、范围、哈希、GT 隔离测试 | synthetic fixture 和验收阈值；真实帧稍后复验 |

Linux 不可用会阻塞“真实数据 Gate”，但不应阻塞以上接口、转换器和 synthetic
测试。任何 synthetic 结果必须明确标注 `synthetic/mock`，不得计入正式数据数量。

## 12. 给各成员的可直接转发摘要

### 给 A

```text
成员 B 已基于 2026-08-02 最新 origin/main 重新整理 Isaac 数据接口。执行适配器
PR #12 已合入，夹爪二值硬件映射也已由代码和测试冻结。请继续冻结三个 P0 决策：
1) 是否维持冻结场景无腕相机、wrist_image=null；
2) Canonical TCP 四元数/rotation-vector 到模型 7D state 的坐标系、旋转表示和单位；
3) OpenVLA/LeRobot 的动作块 N、采样/降采样频率和末尾 padding。
Linux 暂不可用期间建议先放行 synthetic Canonical Episode Gate，不放行真实批采。
```

### 给 C

```text
B 侧已确认：场景 single_bin_static_handoff_v1；三台 RGB 相机为
CAM_A_TOP、CAM_HANDOFF、CAM_B_TOP，均 1280x720；当前无腕相机；动作是
robot_base 下 [dx,dy,dz,dax,day,daz,gripper]，其中 dax/day/daz 共同构成
rotation-vector，不是 droll/dpitch/dyaw 欧拉角增量。请先提供 Canonical
Schema、Recorder 输入接口、1 条标注 synthetic/mock 的 Episode、完整性检查和
Replay 骨架。当前 adapter smoke 已有 robot-base TCP，但尚无三路 RGB/CAS；真实
同步样例待 Linux 恢复后补。
```

### 给 D

```text
B 侧对照了 OpenVLA 输入文档。当前冻结场景不存在 CAM_B_WRIST，仓库配置和测试
要求 wrist_image=null，请先按冻结方案修订，或提交给 A 作架构变更决策。另请明确
state 的精确顺序/坐标/单位、四元数/rotation-vector 到模型 state 的规则、动作块
N、FPS、padding；训练文档中的动作字段应由 droll/dpitch/dyaw 修订为
dax/day/daz，并明确为 robot-base rotation-vector。请按已冻结的动作和二值夹爪
规则提供 Canonical→RLDS/OpenVLA 转换
命令、最小样例和 Loader 成功
输出。可以先用 C 的 synthetic Episode，不必等待 Linux。
```

### 给 E

```text
请提供 π0.5/LeRobot 的目标 features 和最小样例：目录结构、图像/state/action
字段名、shape、dtype、FPS、Episode 边界、动作块 N、padding、转换
命令、Loader 验证及 norm stats 命令。B 侧当前三相机均 1280x720，冻结 MVP 无
腕相机；action 必须保持 `[dx,dy,dz,dax,day,daz,gripper]` 的 robot-base
rotation-vector 语义；夹爪硬件边界为 `<0.5` 闭合、`>=0.5` 打开。可以先用 C 的 synthetic
Canonical Episode，不必等待 Linux。
```

### 给 F

```text
请基于 synthetic Canonical Episode 先给出并实现数据 QA：字段完整性、时间戳单调、
黑帧/重复帧、NaN/Inf、动作范围、图像与 manifest 哈希、GT 在线隔离、转换前后
动作误差和 Episode/Step 数一致性。真实相机帧、掉帧和回放待 Linux 恢复后复验，
synthetic 结果不得计入正式数据数量。
```

## 13. 推荐 Gate 顺序

```text
A 冻结三个剩余 P0 接口决定
        ↓
C 生成 synthetic Canonical Episode
        ↓
D/E 分别完成 RLDS 与 LeRobot Loader smoke
        ↓
F 完成 synthetic QA
        ↓
Linux 恢复：B 完成自建场景键盘、RGB observation 和 1 条真实 Episode
        ↓
C Replay，D/E 重跑转换，F 重跑真实 QA
        ↓
A/F 放行 5 条小样本
        ↓
5 条全通过后再决定是否批量采集
```
