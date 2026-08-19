# XH-202607 工业环境 VLA 智能体

面向“工业环境下物体感知识别与指令交互型智能体研发”比赛的六人协作仓库。
项目目标是在仿真中完成两条相互隔离、共享底层合同的工程链路：

> 预设任务指令 → π0.5 控制 Arm_A 装箱并送至固定交接位 →
> 总控三帧核验并切换控制令牌 → OpenVLA-OFT 控制 Arm_B 搬箱 →
> YOLO 全程旁路留存检测框与 mAP 证据。

冻结基线中总控不做 NLP、复杂度判断或模型选择。π0.5 与 OpenVLA-OFT
在同一个任务中按固定顺序工作：前者只负责 Arm_A，后者只负责 Arm_B；失败时
各自在本阶段使用新鲜观测有界重试，禁止跨角色替换。

当前状态（2026-08-18）：**V2 人工工业场景源码、静态合同、键盘采集、Pink IK
与 Canonical Episode 入口已经合入 `main`；完整 Python CI 已通过。Isaac Sim
可见 GUI、物理、IK、抓取和满载搬运仍需按 V2 Gate 生成正式证据。**

## 当前工业场景口径

当前场景真源是 `single_bin_manual_industrial_v2`，用于人工工业数据采集：

- 两台 Franka：`Arm_A` 负责零件装箱与交接，`Arm_B` 负责料箱搬运；
- 三台固定 RGB 相机：`CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP`；
- 8 个程序化工业零件：4 个轴件、2 个六角螺母、2 把开口扳手；
- A/B/C/D 四个区域各放 2 件，P03/P04 初始倒立；
- 一个 `2×4` 料箱，S11-S24 与 P01-P04/N01-N02/W01-W02 固定映射；
- 料箱中央提梁提供 `BIN_CARRY_TCP`，计划满载质量为 `1.0 kg`；
- 人工键盘动作按 `10 Hz` 写入 Canonical Episode，在线 Observation 禁止 GT。

V2 的配置、构建、采集与验收入口见
[V2 人工工业采集说明](docs/v2-manual-industrial-collection.md)。

`single_bin_pack_handoff_v1` 仍是四 Agent 自动串行闭环的冻结 TaskProfile；其中
P01-P04 和 `2×3` 料箱属于兼容基线，不应被误写为当前 V2 人工采集场景。V2 尚未
宣称已经替换该自动闭环的 TaskProfile、指令或 Supervisor 后置条件。

![中文版：冻结四 Agent 双 VLA 双臂闭环](docs/architecture/assets/four-agent-fixed-dual-vla-architecture-v4-zh.png)

[查看简化 SVG 可编辑版](docs/architecture/assets/four-agent-single-bin-static-handoff-framework-v3.svg)

## 不可变项目基线

- 两份官方 PDF 是需求与验收的唯二官方真源，原字节保存在
  [`docs/official/`](docs/official/)。
- 当前架构图保存在 [`docs/architecture/assets/`](docs/architecture/assets/)；
  原始冻结图与 A-F 分工快照保存在 [`docs/assets/`](docs/assets/)。
- 初版方案 DOCX 仅是可修订参考，保存在 [`docs/source/`](docs/source/)。
- 运行 `python scripts/verify_official_baselines.py` 校验唯二官方 PDF；
  `python scripts/verify_project_frozen_inputs.py` 单独校验两张冻结图和初版 DOCX 快照。
- 冲突、评分和六项提交物的工程化索引见
  [官方需求基线](docs/requirements/official-requirements-baseline.md)。

任何 PR 都不得修改官方 PDF 或弱化每步重观察、失败恢复、仿真初赛等硬要求。

项目架构同时冻结以下边界：

- 四个 Agent：总控、π0.5、OpenVLA-OFT、YOLO；不设置 NLP Agent；
- 当前使用两台机械臂，但共享交接区始终只有一个控制令牌；
- π0.5 固定控制 Arm_A：理解上游预设指令、装入四个零件并移箱至中央交接位；
- Supervisor 连续采集三帧，按至少两票通过后执行
  `A_ONLY → HANDOFF_VERIFY → B_ONLY`；
- OpenVLA-OFT 固定控制 Arm_B：收到 `handoff_ready` 后把满箱搬到
  `FINISHED_01`；
- π0.5 与 OpenVLA-OFT 都必须完成工业场景微调及 base/tuned 对照；
- YOLO 独立保存 bbox、类别、置信度和时延，离线计算 mAP；
- YOLO 是同步调用、失败非门控的评分 sidecar；GT 不得进入任何在线 Agent。

## 现在从这里开始

| 你要做什么 | 入口 |
|---|---|
| 查看 D01-D40 任务、Gate 和降级点 | [40 天逐日计划](docs/project-management/daily-plan.md) |
| 查看 Epic/User Story/Task 分解 | [项目 WBS](docs/project-management/wbs.md) |
| 查看每日 A-F 任务 | [每日任务公告索引](docs/project-management/daily/README.md) |
| 查看每日 09:00 自动发布规则 | [每日任务自动化](docs/project-management/daily-task-automation.md) |
| 学习 clone、分支、提交、PR、冲突处理 | [GitHub 协作指南](docs/project-management/github-collaboration-guide.md) |
| 判断代码、模型、数据和报告应放哪里 | [仓库目录与文件规范](docs/repository-structure.md) |
| 查看团队的 Issue/PR/DoD 规则 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 查看总 Agent 设计 | [Agent 架构文档](docs/architecture/agent-framework.md) |
| 查看当前 V2 工业场景 | [V2 人工工业采集说明](docs/v2-manual-industrial-collection.md) |
| 查看 V2 场景与 V1 自动闭环边界 | [场景与流程总说明](docs/architecture/final-frozen-scene-and-flow.md) |
| 对接 D/E/B/F 服务 | [极详细接口契约](docs/architecture/interface-contracts.md) |
| 采集训练数据并安排 B-F 工作 | [数据采集与五人执行指南](docs/project-management/data-collection-and-five-member-execution-guide.md) |
| 查看真实完成度与评分缺口 | [项目看板](docs/project-management/dashboard.md) |
| 查看风险与回退 | [风险登记册](docs/project-management/risk-register.md) |

## 快速运行总 Agent Mock

要求 Python 3.10+。核心包没有第三方运行时依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python scripts/run_mock_demo.py
python -m unittest discover -s tests -v
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
```

macOS/Linux 使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/run_mock_demo.py
python -m pytest -q
```

演示会依次跑正常闭环、Arm_A 阶段恢复和 Arm_B 阶段恢复三个场景；每个场景
都按顺序调用 π0.5 与 OpenVLA-OFT。它用于验证编排合同，
**不代表真实模型或仿真已经打通**。

开发验收：

```powershell
python -m pip install -e ".[test]"
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python scripts/check_repository_hygiene.py
git diff --check
```

## 核心代码

```text
src/industrial_agent/
├── contracts.py       # Task/Observation/7D Action 合同
├── lifecycle.py       # 冻结任务画像、双臂阶段与独占令牌
├── fsm.py             # 显式状态机
├── executor.py        # 双 VLA 固定角色与独立进程适配器
├── observation.py     # 在线观测白名单和 GT 隔离
├── perception.py      # YOLO 失败非门控 sidecar 合同与 mAP 证据
├── safety.py          # NaN、限幅、工作空间与系统故障
├── verifier.py        # 多帧后置条件核验
├── orchestrator.py    # Arm_A→三帧交接核验→Arm_B 的闭环总循环
└── mock.py            # 无第三方依赖演示环境
```

机器可校验合同位于 [`schemas/`](schemas/)，默认配置位于
[`configs/agent.default.json`](configs/agent.default.json)。

## A-F 冻结职责

| 角色 | 主责 | 关键验收/备份 |
|---|---|---|
| A（队长） | 需求、TaskEnvelope、FSM、令牌/恢复、总集成、答辩 | 验收 B 的动作/安全接口 |
| B | 仿真、Franka/夹爪、控制器、物理、headless | 备份 A 的安全状态机 |
| C | 场景、资产、教师轨迹、canonical 数据、split | 备份 F 的数据 QA |
| D | OpenVLA-OFT 复现、转换、微调、服务 | 备份 E 的服务协议 |
| E | π0.5/openpi、LeRobot、norm stats、训练、服务 | 备份 D 的动作适配 |
| F | YOLO/核验、评测、CI、复现、报告/视频 | 备份 C 的数据 QA |

B-F 姓名与 GitHub 用户名必须由 A 确认后再启用 CODEOWNERS；不得按成员名单
顺序自行猜测映射。

## 项目节奏

- D01：2026-07-25；D40 内部封版：2026-09-02。
- 2026-09-03 至 09-05 仅用于复现、校验、上传和回退。
- 每天 09:00 发布每人一个主任务，17:00 交付，18:00 集成。
- 先 Issue，再短分支，再 Draft PR；禁止直接 push `main`。
- 模型权重、数据、录像、密钥不得进入普通 Git 历史。

完整管理规则见
[项目管理执行指南](docs/project-management/project-management-guide.md)。

## 许可证状态

仓库当前尚未声明开源许可证。A 应在对外分发或允许第三方复用前确认比赛规则、
上游模型/资产许可证与团队授权，再通过独立 PR 添加合适的 `LICENSE`；在此之前
默认不授予外部复制、修改或再分发权利。
