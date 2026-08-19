# 单箱双臂 VLA：数据采集与 B–F 五位成员执行指南

> **2026-08-18 当前执行口径：** 新采集优先使用
> `single_bin_manual_industrial_v2`：8 个工件、A/B/C/D 各 2 件、2×4 料箱，
> 计划通过 `simulation/run_v2_keyboard_collection.py` 人工生成 Canonical Episode；
> 该 B 侧入口及其 V2 场景配置当前尚未进入仓库，补齐前不得正式采集。
> 本文其余四工件、2×3 料箱、冻结双 VLA 指令与 40 天自动闭环安排保留为 V1
> 兼容计划，不应写入 V2 Episode 的场景身份。

## 当前 V2 人工采集基线（优先）

1. 由 B 先提交 V2 场景配置、构建器、GUI 采集入口和场景 Preflight；
2. 在 Isaac Sim 中依次完成静态、HOME、IK 和微动验收；
3. 验收通过后才可使用计划入口
   `python simulation/run_v2_keyboard_collection.py --output-dir <目录>` 采集；
4. 固定槽位映射为 S11=P01、S12=P03、S13=N01、S14=W01、
   S21=P02、S22=P04、S23=N02、S24=W02；
5. 只有实际完成 GUI/物理/IK/抓取/满载搬运并保留证据后，才能将对应状态写为通过。

完整参数与命令见 [V2 人工工业采集说明](../v2-manual-industrial-collection.md)。

> 版本：v1.0
>
> 适用周期：六人 40 天
>
> Owner：A（队长/总集成）
>
> 数据放行：A + F
>
> 场景唯一配置：`simulation/configs/single_bin_scene_v1.json`

## 0. 先看这一页：团队唯一目标

40 天内只完成一个可在 Docker 中复现的最小闭环：

```text
预设自然语言指令
    ↓ 原文，不增加 NLP Agent
π0.5 Agent 控制 Arm_A
    ↓ 完成四个零件装箱、纠正倒放件、满箱放到 HANDOFF_CENTER、退出共享区
Supervisor 先做候选预检，再锁臂并用三张新鲜在线观测和机器人遥测做 2/3 复合投票
    ↓ handoff.candidate_checked 至少 1 次、重试时可为 1..N 次；里程碑为 handoff.verified → handoff.ready
    ↓ 令牌 A_ONLY → HANDOFF_VERIFY → B_ONLY
OpenVLA Agent 控制 Arm_B
    ↓ 抓取同一个满箱并放到 FINISHED_01
Supervisor 用新鲜在线观测和机器人遥测验证终局
    ↓
Supervisor 判定成功、有限恢复或安全停止

每次选定新鲜帧时同步调用 YOLO 评分 sidecar 并关联留证；检测失败只记录，
不阻止 VLA，也不作为交接令牌硬依赖
```

以下内容不进入本项目：

- 不增加 NLP Agent；
- 不增加传送带、滑台或第三机械臂；
- 不做两个 VLA 的动态任务竞标；
- 不把 OpenVLA 改成装零件，也不把 π0.5 改成搬运臂；
- 不扩大到多箱、多成品位或任意工业场景；
- 不把仿真 GT、物体真实坐标直接喂给在线 VLA 或 YOLO。

本指南与 `single_bin_scene_v1.json` 是当前执行依据：两个 VLA 的机械臂和阶段
固定，唯一交接位是 `HANDOFF_CENTER`。

## 1. 冻结对象、节点与任务边界

| 项目 | 冻结内容 |
|---|---|
| 机器人 | 两台 Franka：`Arm_A`、`Arm_B` |
| 主执行 | π0.5 只控制 `Arm_A` |
| 协作执行 | OpenVLA-OFT 只控制 `Arm_B` |
| 检测 | YOLO 对新鲜帧同步调用，独立输出检测框、类别、置信度和离线 mAP 证据；失败非门控 |
| 物料 | P01–P04，共 4 个圆柱件；P02 初始为倒放 |
| 初始分布 | A/B/C/D 区域分别为 2/1/1/0 |
| 料箱 | `Bin_01`，2×3 格，初始位于 `PACK_STATION` |
| 交接 | 固定平面 `HANDOFF_CENTER` |
| 成品位 | 固定平面 `FINISHED_01` |
| RGB 相机 | `CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP` |
| 在线语言 | 比赛演示前冻结的自然语言模板；原文直接进入对应 VLA |
| Supervisor | FSM、令牌、安全、超时、事件、有限重试；不解析自然语言 |

Supervisor 启动任务时读取的是人工预先配置的运行档案，例如
`task_id`、允许的 FSM 阶段和超时，而不是从指令中“猜”生命周期。
π0.5 理解自然语言并执行 A 臂动作；Supervisor 根据冻结 FSM、VLA 事件、
新鲜在线观测、机器人遥测和安全条件推进生命周期。YOLO 结果只做同帧评分
留证，不是调用 VLA 或发放 `B_ONLY` 的硬依赖。

## 2. B–F 固定职责：每个人只守住一条主线

### 2.1 责任表

| 成员 | 唯一主责 | 每天要提交的证据 | 不负责 |
|---|---|---|---|
| B：仿真/控制 | Isaac Sim、双 Franka、夹爪、相机、碰撞、Reset、脚本专家 | 可运行命令、场景配置、日志、短视频、失败 Seed | VLA 训练、标注策略 |
| C：场景/数据 | 资产、Canonical Recorder、随机化、回放、切分、Manifest、备份 | Episode 清单、QA 报告、数据哈希、回放样例 | 决定模型效果、修改动作合同 |
| D：OpenVLA | Canonical→RLDS、OpenVLA-OFT 微调、Arm_B 推理服务 | Loader 输出、训练配置/曲线、权重哈希、接口压测 | Arm_A、Supervisor 生命周期 |
| E：π0.5 | Canonical→LeRobot、π0.5 微调、Arm_A 推理服务、独立 norm stats | Loader 输出、训练配置/曲线、权重哈希、接口压测 | Arm_B、YOLO 指标 |
| F：YOLO/质量 | 自动标注、COCO/YOLO 导出、mAP、数据 QA、测试、证据、材料 | 标注抽检、指标 JSON、测试报告、视频索引、复现记录 | 修改场景物理、替 VLA 规划动作 |

### 2.2 每人本周能直接开工的任务

#### B：先让“场景和动作”可信

1. 用冻结 JSON 生成场景，完成 50 次无异常 Reset。
2. 校验三个 RGB Camera Prim、分辨率、时间戳和视野。
3. 实现两臂的统一 7 维动作执行接口与限幅。
4. 制作脚本专家：
   - A 臂：抓 P01–P04、P02 纠姿、放入 2×3 料箱、把满箱放到
     `HANDOFF_CENTER`、退出；
   - B 臂：等待 durable `handoff.ready`、抓料箱把手、水平搬运到
     `FINISHED_01`、退出。
5. 脚本专家只用于采集示范和测试，不作为最终在线智能体。

#### C：先保证每条数据“可追、可回放、可转换”

1. V1 按 `schemas/canonical-episode.schema.json`，V2 按
   `schemas/canonical-episode-v2.schema.json` 生成 `episode.h5 + structure.json`。
2. 独立记录三路 RGB（30Hz）、双臂 `state_7d`（60Hz）和动作（10Hz）。
3. 在 Episode 开始前分配 Seed 与 Split，禁止采完后按帧随机切分。
4. 实现录制中断清理、完整性检查、回放和 SHA-256 Manifest。
5. 将仿真 GT 写入隔离的 `offline_gt/`；在线目录不得暴露 GT。

#### D：只打通 OpenVLA 的 B 臂链路

1. 用 1 条伪 Episode 验证 RLDS Schema。
2. 用 5 条真实 Canonical Episode 完成转换和 Loader 遍历。
3. 固定输入：B 区 RGB、`wrist_image=null`、B 臂状态、固定协作指令。
4. 固定输出：统一的 7 维动作块。
5. 完成小数据过拟合冒烟后再扩量；不得只凭训练 Loss 宣称有效。
6. 提供 `/health`、`/infer`、超时、取消和错误码。

#### E：只打通 π0.5 的 A 臂链路

1. 用 1 条伪 Episode 验证 LeRobot Schema。
2. 用 5 条真实 Canonical Episode 完成转换和 Loader 遍历。
3. 固定输入：A 区 RGB、`wrist_image=null`、A 臂状态、原始任务指令。
4. 固定输出：统一的 7 维动作块。
5. 只用 Train Split 计算 π0.5 自己的 `norm_stats`，不得与 OpenVLA
   共用。
6. 提供 `/health`、`/infer`、超时、取消和错误码。

#### F：先让 YOLO 证据链独立成立

1. 冻结 5 类：`part_upright`、`part_inverted`、`part_fallen`、
   `bin_box`、`bin_slot`。
2. 从仿真 GT 同时导出 YOLO TXT 和 COCO JSON。
3. 保存原始检测框和离线 COCO prediction JSON，计算
   AP50、AP75、mAP50:95。
4. 对三个相机分别报告类别分布、漏框、错框和推理时延。
5. 验证在线容器没有挂载 COCO GT、实例 ID 或物体真实位姿。
6. 维护固定 Seed 测试、失败录像索引、报告图表和最终材料。

## 3. 前 7 天：最小成功路线

前 7 天只证明“能采、能转、能读、能回放、能在 Docker 冒烟”，
不要求 VLA 收敛，也不承诺 mAP 或闭环成功率。

| 日 | B | C | D | E | F | 当日统一 DoD |
|---|---|---|---|---|---|---|
| D1 冻结 | 复核场景节点、坐标、相机与控制频率 | 定义 Canonical 字段和目录 | 写 RLDS 字段映射 | 写 LeRobot 字段映射 | 冻结 5 类与 QA 表 | 六人确认单位、坐标系、图像、动作和命名 |
| D2 录制一帧 | Reset、相机 RGB、机器人状态可读取 | Recorder 写出 1 条伪 Episode | Loader 读 1 条伪 RLDS | Loader 读 1 条伪 LeRobot | 由一帧 GT 导出 YOLO/COCO | 所有文件可由脚本生成，无手工复制 |
| D3 真轨迹 | 脚本专家各跑 A/B 一条 | 录 A 5 条、B 5 条 | 转换 B 的 5 条 | 转换 A 的 5 条 | 导出至少 20 张检测图 | 无空帧、NaN、越界框；Episode 可回放 |
| D4 小闭环 | 连续 Reset 20 次，修抓取/碰撞 | 固定 Seed、Split、Manifest | 训练/加载小数据 Smoke | 训练/加载小数据 Smoke | YOLO 训练/推理 Smoke | 三个模型链路都能读取一个 Batch 并推理 |
| D5 扩充 S0 | 运行脚本专家批采 | 累计 A 20 条、B 10 条有效成功轨迹 | 检查 RLDS 数量/动作 | 检查 LeRobot 数量/动作 | 累计 200 张有效图 | 转换前后 Episode/Step/指令一一对应 |
| D6 Docker | 提供 Headless/物理参数 | 配置只读数据挂载 | OpenVLA Smoke 服务 | π0.5 Smoke 服务 | YOLO/评测 Smoke 服务 | 同一 Compose 连续启动 3 次 |
| D7 封存 | 提交场景和失败 Seed | 发布 `v0.1-smoke` Manifest | 发布 RLDS 样例和说明 | 发布 LeRobot 样例和说明 | 干净环境复现、归档证据 | 数据/配置/镜像/样例均有 SHA-256 |

### D7 允许的 S0 数量

| 数据产品 | D7 最低数量 | 用途 |
|---|---:|---|
| π0.5 / Arm_A 成功轨迹 | 20 | 验证完整采集与 LeRobot 链 |
| OpenVLA / Arm_B 成功轨迹 | 10 | 验证完整采集与 RLDS 链 |
| YOLO 图像 | 200 | 验证自动标注、训练、推理和 COCO 评测链 |
| 故意失败样例 | 每臂 3 条 | 在外部 QA Registry 记录 `dataset_failure_label` 并验证隔离规则，不直接当模仿目标 |
| 恢复样例 | 每臂 3 条 | 验证从故障后新观测开始记录正确恢复 |

数量不足不是最大风险；字段错误、动作方向错误或数据泄漏才是。
任何一个 Loader 尚未通过时，禁止批量采集。

## 4. D1–D40 阶段安排

| 阶段 | 目标 | B | C | D | E | F | 阶段 Gate |
|---|---|---|---|---|---|---|---|
| D1–D7 管线冒烟 | 三条数据链可运行 | 场景/相机/脚本专家 | Canonical/Recorder/Split | RLDS Smoke | LeRobot Smoke | YOLO/COCO Smoke | 干净环境一键读取小样本 |
| D8–D12 Oracle 与 v0.1 | 稳定采集而非追模型 | 50 局脚本专家，修物理 | 批采、回放、Manifest | 校验 B 臂动作 | 校验 A 臂动作 | 自动 QA、GT 隔离 | 脚本专家成功率达到内部采集要求，回放抽检通过 |
| D13–D18 模型基线 | 两 VLA 均能走统一接口 | 动作回放/限幅 | 冻结 train/val/test v0.5 | OpenVLA 基线 | π0.5 基线 | YOLO 基线和时延 | 两 VLA 各完成一次真实动作冒烟，YOLO 能独立输出框 |
| D19–D24 可训练最小集 | 只补真实失败簇 | 注入可控故障 | 扩量和去重 | OpenVLA 首轮 OFT | π0.5 首轮微调 | mAP 与数据 QA | 达到第 8 节“可训练下限”，未达不得盲目升级规模 |
| D25–D30 完整闭环 | 固定职责串行协作 | 双臂安全/碰撞复测 | 录闭环/恢复集 | B 臂接箱与恢复 | A 臂装箱/交接与恢复 | 固定 Seed 闭环评测 | `A_ONLY → HANDOFF_VERIFY → B_ONLY` 可审计，P0=0 |
| D31–D35 冻结评测 | 不再改测试集 | 第二机/Headless | 冻结数据 v1.0 | 候选权重定版 | 候选权重定版 | ID/OOD、mAP、消融、证据 | 原始 JSONL 可重算全部表格 |
| D36–D40 打包答辩 | Docker 和材料可复现 | 安装/启动复现 | 数据卡、许可、哈希 | 权重/服务/说明 | 权重/服务/说明 | 总测试、视频、报告、包校验 | 干净机连续运行；只修提交阻断 |

### 强制停止扩量的条件

满足任一项即停止采集，先修问题：

- 相机帧与机器人状态时间戳不同步；
- 7 维动作的坐标系、米/弧度或夹爪符号未冻结；
- Episode 不能回放；
- 同一 Seed 被分到不同 Split；
- 在线服务能读到 GT、实例 ID 或物体真实位姿；
- 失败前缀被当成正确动作训练；
- Reset 后物体穿模、机械臂自碰或随机化导致不可达；
- 磁盘没有保留至少一份校验过的备份。

## 5. 每日任务公告模板

A 每天只给每人 1 个主任务。复制以下内容到当日 Issue：

```markdown
## Dxx / 成员 X / 今日唯一主任务

- 目标：
- 输入依赖（PR、数据版本、场景版本）：
- 修改范围（目录/文件）：
- 执行命令：
- 固定 Seed：
- 预期产物：
- Definition of Done：
  - [ ] 命令返回 0
  - [ ] 原始日志已上传
  - [ ] 有可复核的样例/截图/短视频
  - [ ] 数据、配置或权重有 SHA-256
  - [ ] 未引入 GT 在线泄漏
- 截止：当日 18:00
- 阻塞升级：连续阻塞 2 小时即在 Issue @A，并附错误日志和已尝试项
- 实际结果：
- 未完成原因与明日第一动作：
```

每日 18:00 只允许以下三种状态：

- `Done`：DoD 和证据全部存在；
- `Blocked`：有可复现错误、Owner 和下一动作；
- `Carry-over`：未完成且次日仍是唯一主任务。

“研究中、在调参、快好了”不能作为进度证据。

## 6. Canonical Episode：所有模型共用的唯一数据源

### 6.1 外部数据目录

正式数据不进入 GitHub。统一放在团队共享数据盘：

```text
industrial_dataset_root/
├── canonical/
│   ├── arm_a_pi05_v1/
│   │   └── <episode_id>/
│   └── arm_b_openvla_v1/
│       └── <episode_id>/
├── offline_gt/
│   └── <episode_id>/                 # 在线容器禁止挂载
├── exports/
│   ├── lerobot/pi05_v1/
│   ├── rlds/openvla_v1/
│   ├── yolo/industrial_yolo_v1/
│   └── coco/industrial_yolo_v1/
├── splits/
│   ├── split_registry_v1.json
│   └── frozen_test_v1.json
├── manifests/
├── reports/
│   ├── qa/
│   ├── map/
│   └── replay/
└── quarantine/                       # 中断、损坏、越界或待审数据
```

单条 Canonical Episode：

```text
<episode_id>/
├── episode.h5       # 三路 RGB、双臂 state_7d、动作与 valid_mask
└── structure.json   # 元数据、各数据集 shape/dtype/count 与 episode.h5 SHA-256
```

冻结 MVP 只采集以上三台物理 RGB 相机，不创建 `wrist_rgb/`。VLA 接口为兼容统一
Schema 保留 `wrist_image` 键，但本版本在 Episode 元数据中固定写 JSON `null`。

不要把三个相机都强制复制给两个 VLA。Canonical 可以统一保存，转换器只选
对应模型需要的视角，减少显存和训练噪声。

当前结构真源按版本分为 `schemas/canonical-episode.schema.json`（V1）和
`schemas/canonical-episode-v2.schema.json`（V2）；旧的
`meta.json + steps.jsonl` 方案已废止。关节数组、FSM 事件或额外诊断字段如需加入，
必须升级 Canonical Schema 版本，不能临时塞入 HDF5。

### 6.2 `structure.json` 核心字段

| 字段 | 示例/类型 | 负责人 | 说明 |
|---|---|---|---|
| `schema_version` | `"1.0"` | A/C | 变更必须升级版本 |
| `canonical_schema_version` | `"2.0"` | A/C | V2 专用；不得与 V1 版本键并存 |
| `episode_id` | `"train-a-000123"` | C | 全局唯一 |
| `scene_seed` | 整数 | C | Reset、资产和布局根 Seed |
| `task_id` | `"pack_handoff_v1"` | A | 固定运行档案 ID |
| `instruction` | 原始中文字符串 | A/C | 不预解析成物体坐标 |
| `git_sha` | 40 位 Git SHA | C | Recorder 代码版本 |
| `scene_config_sha256` | `sha256:<64hex>` | B/C | 指向冻结场景配置 |
| `frequency_contract` | 120/60/30/10 | B/C | 物理/控制/渲染/模型频率 |
| `padding_policy` | 策略 + 可选长度 | A/C | 默认不 Padding；任何 Padding 必须有 mask |
| V2 `padding_policy` | `none/null` | A/C/E | V2 禁止任何 masked padding 行 |
| `outcome/failure_code` | 终局 + 可空错误码 | C/F | 非成功 Episode 必须有错误码 |
| `wrist_image` | `null` | C | 冻结场景没有腕相机 |
| `offline_gt_included` | `false` | C/F | Canonical/在线数据不含 GT |

### 6.3 HDF5 三条独立时间流

| 流 | 频率 | 核心字段 | 强制规则 |
|---|---:|---|---|
| `cameras/<camera_id>` | 30Hz | RGB、CAS URI、图像 SHA、时间戳、physics tick | 恰好三台相机，1280×720 RGB，三路同 tick/时间戳 |
| `robot_state/<arm_id>` | 60Hz | `state_7d`、时间戳、physics tick | 恰好 Arm_A/Arm_B；`[x,y,z,ax,ay,az,gripper]`，后三维是 rotation-vector |
| `actions` | 10Hz | `action_7d`、臂、执行器、chunk、`valid_mask` | Arm_A→π0.5、Arm_B→OpenVLA；Padding 行不得执行或计入训练损失 |

动作合同固定为：

```text
translation frame = robot_base
translation unit  = metre
rotation unit     = radian
gripper           = normalized [-1, 1]
```

所有 GT 字段，例如实例 ID、真实 6D 物体位姿、无遮挡框和遮挡比例，
只能进入 `offline_gt/<episode_id>/`，不能加入在线 VLA Observation。

## 7. 三条数据线分别采什么

### 7.1 π0.5 / Arm_A

输入：

- 原始自然语言；
- `CAM_A_TOP` RGB；
- `wrist_image=null`；冻结 MVP 不采集 Arm_A wrist RGB；
- Arm_A 关节、TCP 和夹爪状态。

行为目标：

1. 按冻结指令观察工作区中的 P01–P04；
2. 抓取 P01–P04；
3. 对倒放的 P02 完成正常姿态放置；
4. 每格最多一个零件；
5. 抓起满箱并放到 `HANDOFF_CENTER`；
6. 退出共享区并上报 `robot.arm_a.retreated=true`。

必须覆盖的数据类型：

| 类型 | 可训练内容 | 不可训练内容 |
|---|---|---|
| 成功 | 从观察到正确放置的专家动作 | 无 |
| 失败 | 保存观测、状态，并在外部 QA Registry 记录 `dataset_failure_label` 供诊断 | 导致漏抓、掉落、碰撞的错误动作 |
| 恢复 | 从故障后的新观测开始执行正确恢复 | 故障发生前的错误前缀 |

### 7.2 OpenVLA / Arm_B

唯一逐字冻结指令：

> 收到 handoff_ready 后，观察中央交接位，抓稳 Bin_01 并保持水平，将其搬到
> FINISHED_01，松开夹爪并返回 HOME_B。

输入：

- `CAM_B_TOP` 和交接阶段的 `CAM_HANDOFF` RGB；
- `wrist_image=null`；冻结 MVP 不采集 Arm_B wrist RGB；
- Arm_B 关节、TCP、夹爪状态；
- durable `handoff.ready` 是允许 Arm_B 执行的生命周期事件；
- 冻结指令里的 `handoff_ready` 是业务信号文字，不是 `event_type`。

行为目标：

1. 在 `B_ONLY` 前保持不动；
2. 从 `HANDOFF_CENTER` 抓料箱把手；
3. 水平抬升和搬运；
4. 放到 `FINISHED_01`；
5. 退出并上报 `robot.arm_b.retreated=true`。

OpenVLA 数据不得混入 Arm_A 的装零件动作；RLDS 中必须显式记录
`robot_role=arm_b_openvla`。

### 7.3 YOLO / 检测与 mAP

YOLO 从三台 RGB 相机采图，仿真 GT 自动生成标注：

| 类别 ID | 类别 | 采集要求 |
|---:|---|---|
| 0 | `part_upright` | 各区、各相机尺度、机械臂轻度遮挡 |
| 1 | `part_inverted` | P02 初始态和受控变化；顶部/底部必须视觉可区分 |
| 2 | `part_fallen` | 只用受控故障注入产生 |
| 3 | `bin_box` | 初始位、交接位、运输中、成品位 |
| 4 | `bin_slot` | 空格、已占格、部分遮挡；仍标注格口 |

固定 ROI A/B/C/D、`HANDOFF_CENTER` 和 `FINISHED_01` 不作为检测类别；
YOLO 检测与 ROI 用于区域计数、目标检测评分和同帧证据。交接令牌由
Supervisor 的冻结 FSM、新鲜在线观测、时序投票、机器人遥测和安全条件决定，
YOLO 检测成功不是必要条件。

当前在线实现对选定的新鲜帧同步调用一次 YOLO sidecar，以保证
`trace_id + observation_id + image_sha256` 可追溯；这不是“真正异步”
实现。YOLO 成功与否不属于 `A_ONLY → HANDOFF_VERIFY → B_ONLY` 的令牌条件，
也不属于任一 VLA 请求的必填输入。空检测、超时或坏包只产生评分证据缺失记录，
Supervisor 仍依据冻结 FSM、新鲜在线观测、机器人遥测和安全条件推进或停止。

离线评分链必须能独立运行：

```text
RGB + COCO GT
    → YOLO inference
    → COCO predictions.json
    → AP50 / AP75 / mAP50:95
    → 每类 AP + 每相机指标 + 失败样例索引
```

## 8. 数据规模：先下限，后升级

以下是内部项目目标，不是官方承诺，也不保证模型达到某个成功率。

| 数据产品 | S0 冒烟 | 可训练下限（先做到） | 时间充足升级目标 |
|---|---:|---:|---:|
| π0.5 可训练轨迹 | 20 | 160 | 320 |
| OpenVLA 可训练轨迹 | 10 | 100 | 180 |
| YOLO 有效图像 | 200 | 1,500 | 4,000 |
| 冻结闭环评测 Episode | 3 | 100 | 300 |

可训练 VLA 轨迹中建议：

- 65%–75% 标准成功；
- 10%–15% 边缘位置/轻遮挡下的成功；
- 12%–20% 从故障状态开始的正确恢复；
- 失败日志额外保存，不计入“可训练轨迹”数量。

YOLO 图像建议：

- `CAM_A_TOP` 约 60%；
- `CAM_HANDOFF` 约 20%；
- `CAM_B_TOP` 约 20%；
- 困难帧至少 25%，但不能用相邻的近重复帧凑数量；
- 每个类必须同时出现在 Train、Validation、Test，稀有类不足时优先补
  `part_inverted` 和 `part_fallen`。

只在以下全部通过后，从“可训练下限”升级到最终目标：

1. 两个 Loader 可遍历全部数据；
2. 10% Episode 回放抽检通过；
3. Split 无交叉；
4. 动作方向、单位、夹爪符号正确；
5. 首轮训练错误主要来自覆盖不足，而不是数据格式或物理错误。

## 9. 成功、失败与恢复数据的记录规则

### 9.1 终局定义

下表是外部 QA Registry 的数据分类，不替代 Canonical `outcome`。Episode 内的
`outcome` 只能使用 `SUCCEEDED/FAILED/SAFE_STOPPED/SAFE_STOP_FAILED`。

| 标签 | 条件 |
|---|---|
| `success` | 当前 Agent 的固定任务完成，终局空间关系、机器人遥测和安全条件均通过 |
| `failure` | 超时、抓空、掉落、越界、碰撞、料箱倾斜或错误终局，且未恢复 |
| `recovery_success` | 从已保存的故障后状态重新观察，并通过正确动作恢复 |
| `invalid` | 文件缺失、时间戳错乱、动作 NaN、Reset 穿模或录制中断 |

### 9.2 `dataset_failure_label` 最小集合

这些标签属于 `reports/qa/` 下按 `episode_id` 索引的外部 QA Registry，
不是 Canonical Episode v1.0 的 HDF5 或 `structure.json` 字段。

```text
EMPTY_GRASP
PART_DROPPED
PART_WRONG_SLOT
PART_WRONG_ORIENTATION
BIN_OUTSIDE_HANDOFF
ARM_A_NOT_RETREATED
BIN_GRASP_FAILED
BIN_TILTED
BIN_OUTSIDE_FINISHED
ARM_B_NOT_RETREATED
COLLISION_RISK
ACTION_TIMEOUT
OBSERVATION_INVALID
```

### 9.3 恢复 Episode 正确做法

1. 通过固定 Seed 注入一个安全、可复现的故障；
2. 保存故障终点并生成新的 `observation_id`；
3. 新建 Recovery Episode，并在外部 QA Registry 填写 `parent_episode_id`；
4. 专家从故障状态执行正确恢复；
5. 原始失败动作只用于诊断；转换器必须依据 Canonical `valid_mask` 排除无效行；
6. 恢复 Episode 必须与 Parent Episode 属于同一个 Split；
7. 危险碰撞、越区或强制急停只进入安全测试，不进入模仿训练。

## 10. 随机化边界：一次只增加一级

### R0：确定性冒烟（D1–D12）

- 机器人基座、站位、料箱尺寸、格口、任务语言全部固定；
- 使用固定材质、固定相机、固定光照；
- 只改变 `scene_seed` 以验证重置可复现，不改变核心几何。

### R1：可训练最小集（通过 Oracle Gate 后）

| 参数 | 范围 |
|---|---:|
| 零件初始 XY | 基准位置 ±8 mm |
| 零件 yaw | ±5° |
| 料箱初始 XY | ±5 mm |
| 料箱 yaw | ±2° |
| 相机位置 | 每轴 ±5 mm |
| 相机角度 | 每轴 ±0.5° |
| 光照强度 | 基准 ±10% |
| 质量/摩擦 | 基准 ±5% |

### R2：升级集（模型基线稳定后）

| 参数 | 范围 |
|---|---:|
| 零件初始 XY | 基准位置 ±15 mm |
| 零件 yaw | ±15° |
| 料箱初始 XY | ±10 mm |
| 料箱 yaw | ±5° |
| 相机位置 | 每轴 ±10 mm |
| 相机角度 | 每轴 ±1° |
| 光照强度 | 基准 ±20% |
| 质量/摩擦 | 基准 ±10% |
| 轻度遮挡物 | 0–1 个，不进入抓取路径 |

永远不随机化：

- 机械臂基座和 Arm_A/Arm_B 分工；
- `HANDOFF_CENTER` 和 `FINISHED_01` 的语义；
- 2×3 料箱结构；
- 四零件、单箱和 A/B/C/D=2/1/1/0 的 MVP 任务逻辑；
- 任何会导致软工作半径超过 0.65 m 或产生初始碰撞的布局。

B 必须在每个 Reset 后先跑可达性与碰撞预检；失败 Seed 进入
`quarantine`，不能为了数量进入训练集。

## 11. Split 与 Seed：防止数据泄漏

### 11.1 先分组，再采集

以以下组合为不可拆分单元：

```text
QA Registry 中的 scenario_group_id
+ scene_seed
+ asset_variant
+ camera_seed
+ lighting_seed
```

同组的：

- 所有相机帧；
- 所有相邻时刻；
- Parent Failure Episode；
- 派生 Recovery Episode；
- LeRobot、RLDS、YOLO、COCO 导出；

必须进入同一个 Split。

### 11.2 冻结规则

- Train/Validation/Test 目标比例为 70%/15%/15%；
- 采集前由 C 生成 `split_registry_v1.json`；
- F 检查三个 Split 的 Seed 集合交集必须为 0；
- 禁止按图像帧随机切分；
- Test 冻结后，不能因测试结果反复改标注、调阈值或把样本移回 Train；
- 如发现 Test 标注确有错误，必须升数据版本并保留变更记录；
- 最终评测另保留未参与调参的固定 Seed。

## 12. 转换链：Canonical 一次采集，三种输出

```text
Canonical Episode
├── Canonical → LeRobot  → π0.5 / Arm_A
├── Canonical → RLDS     → OpenVLA / Arm_B
└── offline GT + RGB
    ├── → YOLO TXT       → YOLO training
    └── → COCO JSON      → mAP evaluation
```

### 12.1 LeRobot 放行检查

- Episode、Step、图像、语言、状态和动作数量一致；
- `robot_role` 只能是 `arm_a_pi05`；
- 动作仍为统一 7 维合同；
- Train Split 独立计算并保存 π0.5 `norm_stats`；
- 随机抽 10 个 Step，转换前后动作误差小于数值序列化容差；
- Loader 可从头到尾遍历，无在线下载。

### 12.2 RLDS 放行检查

- Episode 边界、`is_first/is_last/is_terminal` 正确；
- `robot_role` 只能是 `arm_b_openvla`；
- 指令、图像、状态与动作时间对齐；
- OpenVLA 使用自己的预处理配置和归一化统计；
- 随机抽 10 个 Step，能反查 Canonical `episode_id/sequence_id`；
- Loader 可从头到尾遍历，无在线下载。

### 12.3 YOLO/COCO 放行检查

- 类别 ID 在所有版本中稳定；
- 框满足 `0 ≤ x < width`、`0 ≤ y < height`、宽高大于 0；
- YOLO TXT 与 COCO JSON 来自同一份 GT，而不是分别手工标；
- `image_id`、`annotation_id` 全局唯一且可由 Manifest 追溯；
- Prediction JSON 与 GT JSON 严格分离；
- mAP 命令、依赖、模型权重和结果 JSON 全部归档。

## 13. 数据 QA 门槛

| 检查 | S0/MVP 门槛 | Owner |
|---|---:|---|
| 必填文件/字段完整率 | 100% | C |
| NaN/Inf/动作维度错误 | 0 | C/F |
| 图像黑帧、解码失败 | 0 个有效 Episode | B/C |
| RGB 与状态时间差 | 不超过一个渲染周期（约 34 ms） | B/C |
| Step 索引连续 | 100% | C |
| Split Seed 交集 | 0 | C/F |
| GT 进入在线挂载 | 0 | F |
| YOLO 框越界/零面积 | 0 | F |
| COCO/YOLO 人工可视抽检 | 至少 10%，框错误率 <2% | F |
| Episode 回放抽检 | 至少 10 条或总量 10%，取较大者 | B/C/F |
| 转换前后 ID/Step 数一致 | 100% | C/D/E |
| 数据哈希覆盖 | 100% 发布文件 | C/F |

首版模型指标只能写成“测得结果”，不能在训练前承诺。YOLO 的内部参考目标
可使用 AP50、AP75 和 mAP50:95，但最终报告必须同时给出每类 AP、样本量、
固定 Split、置信区间或多 Seed 波动，不能只截一张最好结果。

## 14. GitHub 协作规则

### 14.1 GitHub 放什么

允许提交：

- Schema、类别表、Split 清单和数据卡；
- 场景、Recorder、转换器、训练和评测代码；
- 配置、依赖锁文件、测试和少量脱敏 Fixture；
- 数据/权重/Docker 的 SHA-256、下载或装载说明；
- 指标 JSON、汇总 CSV、日志摘要和低码率证据视频链接。

禁止提交：

- 正式 RGB/轨迹全量数据；
- 大模型权重和大体积 Docker tar；
- `.env`、Token、账号、绝对个人路径；
- 临时缓存、训练中间碎片和无法解释的二进制文件。

### 14.2 一项任务一个 Issue、一个分支、一个 PR

分支示例：

```text
member-b/isaac-reset-camera
member-c/canonical-recorder
member-d/openvla-rlds-loader
member-e/pi05-lerobot-loader
member-f/yolo-coco-export
```

提交示例：

```text
feat(sim): add deterministic dual-arm reset
feat(data): record canonical episode metadata
feat(openvla): add RLDS loader smoke test
feat(pi05): add LeRobot action mapping
test(yolo): validate COCO bounding boxes
docs(data): publish v0.1 dataset manifest
```

PR 必须包含：

```markdown
- 关联 Issue：
- 修改范围：
- 本地执行命令：
- 测试结果：
- 数据/场景/权重版本：
- 固定 Seed：
- 截图/日志/样例：
- 风险与回退：
- [ ] 未提交数据、权重、密钥或绝对路径
- [ ] 未更改冻结场景和四 Agent 职责
```

团队规则：

1. 禁止直接 Push 到 `main`；
2. PR 尽量控制在一个可验收目标内；
3. B/C 接口变更必须先通知 A、D、E、F；
4. Schema、动作合同或类别 ID 变更必须升版本，不能静默修改；
5. F 只按提交的命令和证据复现，不接受“我电脑上能跑”；
6. 合并后由产物 Owner 更新 Manifest 和 SHA-256；
7. 大文件失败时不要反复强推 Git，应转移到团队数据盘。

## 15. 依赖关系和集成顺序

```text
A 冻结动作/事件/数据合同
    ↓
B 场景 + 相机 + 控制 + 脚本专家
    ↓
C Canonical Recorder + Split + Replay
    ├──→ E LeRobot → π0.5 / Arm_A
    ├──→ D RLDS → OpenVLA / Arm_B
    └──→ F YOLO/COCO → mAP/评分证据
                   ↓
D + E 提供稳定服务，F 提供验证结果
                   ↓
A 集成 Supervisor FSM 和 Docker 闭环
```

| 交付方 | 接收方 | 必须先交付 |
|---|---|---|
| A | B/C/D/E/F | 动作合同、事件名、错误码、FSM 阶段 |
| B | C | 可重置场景、同步相机/状态、脚本动作和 Seed |
| C | D/E/F | Canonical Schema、真实样例、Split、Manifest |
| D/E | A | 统一 `/health`、`/infer`、取消、超时、模型身份 |
| F | A/全员 | QA Gate、mAP、固定 Seed 评测和证据索引；YOLO 结果不控制令牌 |

任何人不能绕开依赖自行发明第二套字段或动作格式。

## 16. 每位成员的最终交付物与 Definition of Done

### B / 仿真平台

- 冻结场景 USD/JSON 和资产清单；
- 两台 Franka、三台 RGB 相机、夹爪和统一控制器；
- Headless Reset/Run 命令；
- 脚本专家与故障注入；
- 50 次 Reset、碰撞/可达性和固定 Seed 报告。

DoD：干净环境连续 3 次启动；关键路径可执行；Reset 无穿模/初始碰撞；
相机、状态、动作时间同步；A/B 臂在错误令牌下不能进入共享区。

### C / 场景资产与数据

- Canonical Schema、Recorder、Replay；
- Split Registry、Dataset Card、Manifest 和 SHA-256；
- R0/R1/R2 随机化配置；
- 自动 QA 和隔离目录；
- 可供 D/E/F 使用的冻结数据版本。

DoD：有效 Episode 文件完整；回放抽检通过；Split 无泄漏；转换后可追溯；
两份校验过的备份存在。

### D / OpenVLA

- Canonical→RLDS 转换器与测试；
- OpenVLA-OFT 训练配置、日志、权重哈希；
- Arm_B 服务、接口文档和压测；
- Base/Tuned 在同一冻结集上的结果；
- 失败样例与模型限制。

DoD：只控制 Arm_B；从 durable `handoff.ready` 开始；输出统一动作；服务可取消、
超时和报告模型身份；Docker 离线加载成功。

### E / π0.5

- Canonical→LeRobot 转换器与测试；
- π0.5 训练配置、独立 norm stats、日志和权重哈希；
- Arm_A 服务、接口文档和压测；
- Base/Tuned 在同一冻结集上的结果；
- 失败样例与模型限制。

DoD：只控制 Arm_A；原始指令直接进入模型；完成装箱/交接职责；输出统一
动作；Docker 离线加载成功。

### F / YOLO、测试与材料

- 5 类类别表、YOLO/COCO Exporter 和测试；
- YOLO 权重、Prediction JSON、mAP 原始结果；
- 每类/每相机错误分析；
- 数据 QA、固定 Seed 闭环评测；
- 视频、图表、复现记录和最终包清单。

DoD：YOLO 能独立保存检测框并计算 mAP；当前同步 sidecar 的时延单独记录；
空检测、超时和坏包不阻止 VLA，也不控制令牌；在线无 GT；所有表可从原始
JSON/CSV 重算；第二台机器能按 README 复现。

## 17. 主要风险与不改架构的降级方案

| 风险 | 早期信号 | 处理 | 允许的降级 |
|---|---|---|---|
| 仿真抓取不稳定 | 脚本专家成功率低、箱体抖动 | B 先修碰撞体、质量、摩擦、夹爪和路点 | 固定 R0 布局，保留双臂与单箱 |
| 动作合同错误 | 两模型方向相反、夹爪开合颠倒 | 制作 Golden Episode，逐轴回放 | 暂停扩量，只保留 5 条 Smoke |
| 数据管线返工 | Loader 缺帧或步数不一致 | 先冻结 Schema 并加契约测试 | 回退上一个数据版本，不手工修大数据 |
| GPU/时间不足 | OOM、训练速度过慢 | LoRA/OFT、减小 Batch、串行占用 GPU | 先完成可训练下限，不追 320/180 |
| VLA 效果不足 | 固定验证集无提升 | 分析前三失败簇，只补针对性数据 | 固定单布局/单指令模板；两个 VLA 固定职责仍保留 |
| OpenVLA 搬箱困难 | 抓把手失败、箱体倾斜 | 优化把手碰撞体、抓取示范和安全限幅 | 减少随机化，保持 B 臂 VLA 搬同一料箱 |
| YOLO mAP 低 | 倒放/横倒混淆 | 增强顶部/底部视觉差异，补稀有类 | 保持 5 类，先报告真实基线，不删除难类 |
| 测试泄漏 | mAP 异常高、相邻帧跨集 | 按 Episode/Seed 重建 Split | 废弃泄漏版本并升数据版本 |
| Docker 离线失败 | 启动时下载权重/依赖 | 提前缓存并记录许可、哈希和路径 | 去掉非必要可视化，不能去掉四 Agent |
| 双臂碰撞 | 共享区同时出现两臂 | 强制令牌、全臂碰撞检查和退出确认 | 串行执行；不改成第三机械臂或传送带 |

降级的原则是减随机化、减数据规模、减增强功能，不改变：

```text
Supervisor + π0.5 + OpenVLA + YOLO
+ 双 Franka
+ 四零件
+ 单个 2×3 料箱
+ 固定 HANDOFF_CENTER
```

## 18. A 每晚的放行清单

```markdown
- [ ] B 的场景/控制变更有固定 Seed、日志和可回退配置
- [ ] C 的新 Episode 通过 Schema、时间戳、回放、Split 和哈希检查
- [ ] D 的 RLDS 可追溯到 Canonical Episode
- [ ] E 的 LeRobot 可追溯到 Canonical Episode
- [ ] F 的 YOLO/COCO 来自同一 GT，在线容器无 GT
- [ ] YOLO 空检测、超时或坏包没有阻止 VLA，也没有控制交接令牌
- [ ] 两个 VLA 没有交换固定职责
- [ ] HANDOFF token 顺序没有改变
- [ ] 今日 PR 有测试和复现命令
- [ ] 未完成项有 Owner、外部 QA Registry 记录和明日第一动作
```

只要这份清单每天执行，团队就不会因为“又出现一种新方案”而失去主线。
