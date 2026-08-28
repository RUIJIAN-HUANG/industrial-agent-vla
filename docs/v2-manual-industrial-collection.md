# V2 人工工业零件采集

> 场景 ID：`single_bin_manual_industrial_v2`
>
> 当前状态：V2 场景配置、构建器、GUI/物理/IK 验收入口，以及 Canonical V2
> Recorder、Reader 和转换 Preflight 均已进入仓库。正式采集仍须在目标 Isaac Sim
> 环境完成全部验收并保留证据。
>
> 配置真源：`simulation/configs/single_bin_scene_v2.json`

V2 是唯一正式场景。V1 配置与脚本已废除并仅保留为历史回归材料，不得用于
部署、演示、评测或新数据采集。

## 当前实现边界

场景与采集入口：

- `simulation/configs/single_bin_scene_v2.json`：V2 场景机器真源；
- `simulation/v2_scene_contract.py`：数量、坐标、槽位、质量和相机合同；
- `simulation/v2_industrial_assets.py`：轴件、螺母和扳手程序化资产；
- `simulation/single_bin_scene_v2_builder.py`：机器人、相机、料箱和工件构建；
- `simulation/run_v2_gui_scene_acceptance.py`：可见 GUI 场景验收；
- `simulation/run_v2_home_acceptance.py`：双臂 HOME 验收；
- `simulation/run_v2_ik_reachability_acceptance.py`：IK 可达性验收；
- `simulation/run_v2_dual_arm_micro_motion_acceptance.py`：双臂微动作验收；
- `simulation/run_v2_keyboard_collection.py`：人工键盘采集入口；
- `simulation/v2_collection_preflight.py`：正式采集预检。

数据链路：

- `schemas/canonical-episode-v2.schema.json`：V2 落盘合同；
- `src/industrial_agent/data/recorder_v2.py`：无 padding 的 V2 Recorder；
- `scripts/pi05/canonical_v2.py`：Schema、HDF5、SHA 和数值级 Reader；
- `simulation/v2_collection_recorder.py`：Isaac GUI 入口使用的同步写入边界；
- `scripts/pi05/convert_openpi_v2.py`：10 步完整窗口和 N−9 Preflight/转换。

代码存在不等于目标环境验收通过。没有 GUI、HOME、IK、碰撞、抓取和满载搬运
证据时，不得开始正式采集。

## 冻结指令目录

以下五条是当前冻结的用户指令目录，机器真源为
`configs/mvp-instruction-options.json`。用户选择右侧自然语言后，系统解析出左侧
`task_id`，并将该 `task_id` 发送给总控 Agent。界面、采集后台和训练数据必须使用
这里的精确文本，不得自行改写、补充或删除：

| task_id（发送给总控 Agent） | 用户选择的指令 |
|---|---|
| `P01_TO_S11` | 请将轴件 P01 放置到料箱的 S11 格子中。 |
| `W01_TO_S14` | 请将扳手 W01 放置到料箱的 S14 格子中。 |
| `P03_UPRIGHT_TO_S12` | 请将倒立的轴件 P03 翻正后，放置到料箱的 S12 格子中。 |
| `BIN01_TO_FINISHED01` | 请将料箱 Bin_01 搬运到成品区 FINISHED_01。 |
| `PACK_ALL_AND_FINISH` | 请将所有零件按指定位置装入料箱 Bin_01，再将料箱 Bin_01 搬运到成品区 FINISHED_01。 |

当前 Canonical V2 正式采集入口和 Episode Schema 已冻结
`P01_TO_S11` 与 `W01_TO_S14`；其余三条先完成指令冻结，必须在各自任务合同和采集入口完成后，
才能作为对应任务的正式训练数据采集。不得把它们伪装成 P01 Episode。

## 场景组成

- 两台 Franka：`Arm_A`、`Arm_B`，均有显式 HOME；
- 三台 `1280×720`、82° HFOV 固定 RGB 相机：`CAM_A_TOP`、
  `CAM_HANDOFF`、`CAM_B_TOP`；
- 四个轴件：P01/P02 正立，P03/P04 倒立；
- 两颗带可见通孔的简化六角螺母：N01/N02；
- 两把带平行手柄和开口端的简化扳手：W01/W02；
- A/B/C/D 四个区域各 2 件；
- 一个 `0.30×0.22×0.09 m` 的 `2×4` 料箱；
- 固定 S11-S24 配方映射和中央提梁 `BIN_CARRY_TCP`；
- 计划满载质量 `1.0 kg`，计划重心相对提梁投影误差 `3 mm`；
- 在线 Observation/Canonical 字段禁止 GT，GT 只能进入离线目录。

## 固定槽位映射

| 槽位 | 零件 | 类型 | 槽位 | 零件 | 类型 |
|---|---|---|---|---|---|
| S11 | P01 | 轴件 | S21 | P02 | 轴件 |
| S12 | P03 | 轴件 | S22 | P04 | 轴件 |
| S13 | N01 | 螺母 | S23 | N02 | 螺母 |
| S14 | W01 | 扳手 | S24 | W02 | 扳手 |

## 入口与用途

| 入口 | 用途 | 是否可作为正式场景通过证据 |
|---|---|---|
| `run_v2_scene_acceptance.py` | 无 Isaac Sim 的静态合同、资产和质量预算检查 | 否 |
| `build_single_bin_scene_v2.py` | 生成 V2 USD，可用于诊断 | 否 |
| `run_v2_gui_scene_acceptance.py` | 可见 GUI 构建、保存 USD、三相机图和总览图 | 是，限场景外观 |
| `run_v2_home_acceptance.py` | 两臂 HOME 与控制器检查 | 是，限 HOME |
| `run_v2_ik_reachability_acceptance.py` | V2 目标位 IK 可达性 | 是，限 IK |
| `run_v2_dual_arm_micro_motion_acceptance.py` | 双臂微动作和共享区安全门禁 | 是，限微动作 |
| `run_v2_keyboard_collection.py` | 可见 GUI 人工键盘采集一个 Canonical Episode | 仅在预检与数据 QA 全通过后 |

## 必须遵守的验收顺序

1. 运行 V2 离线契约与静态检查：

   ```powershell
   python simulation\run_v2_scene_acceptance.py `
     --evidence-dir artifacts\v2\static
   ```

2. 使用 Isaac Sim 可见 GUI 构建场景，保存 USD、三路相机图和总览图。
3. 验证两臂显式 HOME、IK、碰撞与交接互锁。
4. 依次练习正立轴件、倒立轴件纠正、螺母和扳手。
5. 验证空箱、满箱和 20 次满载搬运。
6. 最后执行正式 Canonical Episode 采集。

在 GUI 验收完成前，不得把静态 PASS 描述为场景正式通过；在完整数据校验完成前，
不得把练习数据标记为可训练数据。

## Canonical→LeRobot 数据 Preflight

从成功母 Episode 生成回放变体时，先按
[V2 回放轨迹批处理生成](v2-replay-batch-generation.md)生成配置、执行并完成哈希/去重
验收；不得直接把未 finalization 的回放目录交给转换器。

获得至少一条正式 V2 Episode 和经过 SHA 校验的 Split Registry 后，先运行只读检查：

```powershell
python scripts\pi05\convert_openpi_v2.py `
  --data-dir <CANONICAL_V2_ROOT> `
  --split-registry <SPLIT_REGISTRY_JSON> `
  --preflight-only
```

每条 Episode 必须包含连续 10 Hz 动作，N 条动作只生成 N−9 个完整 `[10,7]`
窗口。N<10、缺 tick、padding、NaN/Inf 或错误身份一律拒绝。真实 LeRobot 转换还
要求在固定训练环境安装 LeRobot；当前普通 CI 环境不包含该依赖。

## P01_TO_S11 离线成功门禁

Episode 只有同时满足以下三个离线 GT 条件，才允许将 Canonical
`metadata.outcome` 写为 `SUCCEEDED`：

1. P01 的有向本体轴与料箱局部竖直轴的夹角不超过 15°；
2. 终端保持期间执行 3 个不同 physics tick 的新鲜 GT 观测，完整 GT
   判定至少 2 个通过；
3. 真实执行 10 个连续 100 ms 保持动作，保持跨度至少 1.0 s，P01 参考点
   相对保持起点的最大位置漂移不超过 1 mm。

判定实现位于 `simulation/v2_terminal_success.py`，Isaac 读取位于
`simulation/offline_gt.py`。详细角度、位置、时间戳和投票只写入
`offline_gt/p01_terminal_success.json`；Canonical 始终保持
`offline_gt_included=false`。失败时使用明确的 P01 GT failure code，不能降级
为 `SUCCEEDED`。

## 状态声明规则

- 静态 `PASS` 只表示 JSON、程序化资产、槽位、质量和相机合同一致；
- 没有四张 GUI 证据时，不得宣称场景视觉验收通过；
- 没有 HOME、IK、碰撞和搬运证据时，不得宣称机器人执行通过；
- 练习 Episode 默认不可训练；只有预检、终局、回放、GT 隔离和数据 QA 全通过，
  才能把 Episode 标记为训练可用；
- V2 当前是人工采集链路，不得把它描述为已完成的八件全自动 Supervisor 闭环。
