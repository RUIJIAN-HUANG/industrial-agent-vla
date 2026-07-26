# 单箱四零件 + 中央固定交接位：当前冻结场景 v2

> 日期：2026-07-26
> 状态：当前有效，替代此前“三箱 + 传送带”场景草案
> 架构不变：Supervisor、π0.5、OpenVLA、YOLO 四个 Agent；双机械臂；无 NLP Agent

## 1. 场景冻结

| 项目 | 当前定义 |
|---|---|
| 零件 | 4 个，A/B/C/D 区数量为 `2/1/1/0` |
| 姿态 | A 区 1 个正常、1 个倒放；其余正常 |
| 料箱 | 仅 1 个小型 2×3 工业料箱 |
| 装箱配方 | `BIN_01=[P01,P02,P03,P04]` |
| 箱内最终状态 | 4/4 配方完成，2×3 格口保留 2 个空格 |
| 初始箱位 | 蓝色工作区下方中间的 `PACK_STATION`，不靠左边缘 |
| 中央交接 | 固定平面 `HANDOFF_CENTER`，无传送带、滑台或运动机构 |
| 成品区 | 仅 1 个固定成品位 `FINISHED_01` |
| 调度 | A 放箱并完全退出后，B 才能进入共享交接区 |

## 2. 当前图片

- 仿真节点与坐标标注图：[`assets/isaac-sim-single-bin-annotated-platform-v1.png`](assets/isaac-sim-single-bin-annotated-platform-v1.png)
- 场景效果图：[`assets/isaac-sim-single-bin-pack-position-v6.png`](assets/isaac-sim-single-bin-pack-position-v6.png)
- 精确俯视图：[`assets/isaac-sim-single-bin-static-handoff-layout-v2.png`](assets/isaac-sim-single-bin-static-handoff-layout-v2.png)
- 俯视图 SVG：[`assets/isaac-sim-single-bin-static-handoff-layout-v2.svg`](assets/isaac-sim-single-bin-static-handoff-layout-v2.svg)
- 同步框架图：[`assets/four-agent-single-bin-static-handoff-framework-v3.png`](assets/four-agent-single-bin-static-handoff-framework-v3.png)
- 框架图 SVG：[`assets/four-agent-single-bin-static-handoff-framework-v3.svg`](assets/four-agent-single-bin-static-handoff-framework-v3.svg)

## 3. Isaac Sim 推荐坐标

坐标系：

```text
world 原点：HANDOFF_CENTER 在台面的投影
+X：Arm A → Arm B
+Y：工作台前侧 → 后侧
+Z：向上
Stage：Z-up，metersPerUnit=1
台面高度：z=0.750 m
```

| Prim / ROI | 中心 `(x,y,z)` m | 尺寸或说明 |
|---|---:|---|
| `/World/Workcell/Table` | `(0.00,0.00,0.725)` | `2.30×1.10×0.05 m` |
| `/World/Robots/Arm_A` | `(-0.55,-0.30,0.750)` | Franka，面向 `+Y` |
| `/World/Robots/Arm_B` | `(+0.50,-0.30,0.750)` | Franka，面向 `+Y` |
| `/World/Stations/Pack` | `(-0.35,-0.15,0.750)` | 蓝区下方中间的初始装箱位 |
| `/World/Bins/Bin_01` | `(-0.35,-0.15,0.785)` | 初始位于 `PACK_STATION`，`0.18×0.12×0.07 m` |
| `/World/Handoff/Center` | `(0.00,0.00,0.785)` | 满箱交接 ROI，建议 `0.26×0.20 m` |
| `/World/Finished/Slot_01` | `(+0.70,+0.10,0.785)` | 唯一成品位 |

按上述坐标计算的水平距离：

| 路径 | 距离 | 判断 |
|---|---:|---|
| A 基座 → 初始料箱 `PACK_STATION` | `0.25 m` | 充足 |
| A 基座 → 最远零件 | `0.61 m` | 充足 |
| A 基座 → `HANDOFF_CENTER` | `0.63 m` | 充足 |
| B 基座 → `HANDOFF_CENTER` | `0.58 m` | 充足 |
| B 基座 → `FINISHED_01` | `0.45 m` | 充足 |

全部低于 V0 规定的 `0.65 m` 常用工作半径软上限，也低于 Franka 约 `0.855 m`
的最大工作半径。正式冻结前仍需在 Isaac Sim 中用 IK 对抓取前位、抓取位、抬升位
和放置位逐点验证，不能只用平面距离代替可达性测试。

零件：

| 零件 | 区域 | 初始中心 `(x,y,z)` m | 姿态 |
|---|---|---:|---|
| `P01` | A | `(-0.90,+0.20,0.772)` | 正常 |
| `P02` | A | `(-0.80,+0.20,0.772)` | 倒放，`roll=π` |
| `P03` | B | `(-0.60,+0.20,0.772)` | 正常 |
| `P04` | C | `(-0.85,0.00,0.772)` | 正常 |
| D 区 | D | 无 | 空区域负样本 |

固定相机：

| 相机 | 位置 `(x,y,z)` m | Look-at | 使用者 |
|---|---:|---:|---|
| `CAM_A_TOP` | `(-0.65,-0.15,1.50)` | `(-0.65,+0.12,0.75)` | π0.5、YOLO |
| `CAM_HANDOFF` | `(0.00,-0.45,1.18)` | `(0.00,0.00,0.785)` | YOLO、Verifier |
| `CAM_B_TOP` | `(+0.60,-0.18,1.45)` | `(+0.48,+0.18,0.75)` | OpenVLA、YOLO |

统一使用 `1280×720`，HFOV 建议 `65°～72°`。

## 4. 共享交接区安全规则

```text
HANDOFF_ZONE:
x ∈ [-0.16,+0.16]
y ∈ [-0.12,+0.12]
z ∈ [0.74,1.15]
```

普通工作时：

```text
Arm A：x ≤ -0.16
Arm B：x ≥ +0.16
```

只有持有交接令牌的机械臂可以进入 `HANDOFF_ZONE`，并且必须检查整条机械臂碰撞体，
不能只检查 TCP。

令牌顺序固定为：

```text
A_ONLY
→ HANDOFF_VERIFY
→ B_ONLY
```

禁止直接从 `A_ONLY` 切到 `B_ONLY`。

`HANDOFF_VERIFY` 期间：

1. 两臂新动作均锁定；
2. A 臂旧 ActionChunk 已取消；
3. A 臂连续 3 个控制周期满足 `TCP x≤-0.30`；
4. A 臂全部关节速度 `<0.02 rad/s`；
5. A 夹爪已张开；
6. 料箱在交接 ROI 内连续 3 个新帧稳定；
7. 料箱帧间中心移动 `<3 px`；
8. 料箱偏航误差 `<5°`。

全部通过后，Supervisor 才发布 `handoff_ready` 并将令牌切为 `B_ONLY`。

## 5. 两个 VLA 的指令

π0.5 主指令：

```text
将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。
装箱完成后，将料箱放到中央交接位并返回 HOME_A。
失败时重新观察后继续。
```

OpenVLA 固定协作指令：

```text
收到 handoff_ready 后，观察中央交接位，抓稳 Bin_01 并保持水平，
将其搬到 FINISHED_01，松开夹爪并返回 HOME_B。
```

Supervisor 不解析上述语言，只根据冻结 TaskProfile 管理生命周期。

## 6. 单箱闭环

```text
RESET
→ OBSERVE_A
→ π0.5 依据完整 A 区图像处理 P01–P04
→ PICK / VERIFY / PLACE
→ 倒放件纠姿
→ 继续装入 B、C 区零件
→ BIN_READY（配方 4/4）
→ A 将箱放到 HANDOFF_CENTER
→ A 退出并回 HOME_A
→ HANDOFF_VERIFY
→ handoff_ready + B_ONLY
→ OpenVLA 控制 B 抓箱
→ 搬到 FINISHED_01
→ VERIFY_B
→ B 回 HOME_B
→ TASK_SUCCEEDED
```

## 7. 最终验收

| 验收项 | 标准 |
|---|---|
| 视觉任务 | π0.5 按冻结指令处理 P01～P04；Supervisor 不解析语言 |
| 装箱 | P01～P04 全部进入唯一料箱 |
| 姿态 | P02 由倒放变为正常 |
| 交接 | A 放箱后完全退出；B 之后才进入中央区 |
| 搬运 | B 将同一料箱放到 `FINISHED_01` |
| 闭环 | 至少一次失败能刷新观测并有限恢复 |
| YOLO | 对冻结完整类别表保存全部 bbox、类别、置信度与时延；不控制 VLA |
| mAP | 冻结 GT 仅在离线评测器中计算 |
| 安全 | 无双臂同时进入共享区、无碰撞、无无限重试 |
| 复现 | Docker 固定 seed 连续 3 次成功 |
