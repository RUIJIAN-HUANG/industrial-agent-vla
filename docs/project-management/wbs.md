# XH-202607 工作分解结构（WBS）

> 基线：2026-07-24
> 执行周期：D01 2026-07-25 至 D40 2026-09-02
> 原则：每个 Story 必须拆成单人 0.5-1 天 Task，并以 Issue/PR/日志验收。

## 1. Epic 总览

| Epic | 目标 | Owner | 周期 | Gate | 当前状态 |
|---|---|---|---|---|---|
| E00 基线与协作 | 冻结需求、评分、版本、GitHub 和证据规范 | A/F | D01-D02 | G0 前置 | Partial |
| E10 仿真与机器人 | 唯一仿真平台、Franka/夹爪、相机、控制与安全 | B | D01-D18 | G0/G1/G2 | No evidence |
| E20 场景与数据 | 三任务族、工业资产、canonical episode、转换与数据卡 | C/F | D01-D20 | G2 | No evidence |
| E30 总 Agent | 固定 TaskProfile、生命周期/FSM、安全门、同角色重试、交接令牌与可观测性 | A | D01-D30 | G3/G5 | Partial（Mock） |
| E40 OpenVLA-OFT | 官方复现、OFT 数据、微调、服务、统一动作 | D | D01-D33 | G3/G4 | No evidence |
| E50 π0.5/openpi | 官方复现、LeRobot、norm stats、训练、服务 | E | D01-D33 | G3/G4 | No evidence |
| E60 感知与核验 | YOLO/开放域感知、在线后置条件、评测与 CI | F/C | D01-D35 | G2/G5/G6 | Partial（合同） |
| E70 集成与实验 | 三任务闭环、固定双 VLA 协作、同角色恢复、消融、OOD 与迁移 | A/F | D17-D35 | G3-G6 | No evidence |
| E80 报告与提交 | 复现、技术报告、视频、答辩、打包和双备份 | F/A | D30-D40 | G6/G7 | No evidence |

`Partial（Mock/合同）` 只表示结构和单元测试存在，不表示真实模型、仿真或比赛
指标已经完成。

## 2. Story 与验收

| Story | User Story / 结果 | Owner / Support | 截止 | 交付 | Definition of Done |
|---|---|---|---|---|---|
| GOV-01 | 作为队长，我要把官方条款映射到需求/评分/提交物，使任何范围决定可追溯 | A/F | D01 | `official-requirements-baseline.md`、Issue 标签 | 2 个官方源和 3 个项目冻结快照分别校验通过；所有 P0 需求有 owner、证据类型和 Gate |
| GOV-02 | 作为团队成员，我要用同一 Git 流程交付，避免 main 污染和文件丢失 | A/F/全员 | D01 | Guide、CONTRIBUTING、模板、CI | 六人可 clone/push；每人一个 Draft PR；main 禁止直推 |
| SIM-01 | 作为仿真人员，我要证明固定 Isaac Sim 平台可稳定 headless 运行 | B/F | D02 | 版本清单、1000 步日志、3 次重启、相机样本 | G0 通过；失败则回到已验证镜像并删减渲染/材质，不引入第二平台 |
| SIM-02 | 作为策略开发者，我需要稳定的 L0 教师环境，以便先验证物理和控制 | B/C/F | D06 | Franka/夹爪/圆柱/料箱场景、50 局报告 | 成功率 >=90%；无穿模/爆炸/越界；失败均有错误码 |
| SIM-03 | 作为总 Agent，我需要 `observe/step/safe_stop` 环境接口 | B/A/F | D10 | 适配器、坐标/单位、相机同步、安全故障注入 | 每步新 observation_id；safety 完整；safe_stop 清控制器缓冲 |
| DATA-01 | 作为评测方，我要可生成指定格、最多区域、装满搬运三任务族 | C/A/F | D08 | task catalog、语言模板、100 seed smoke | 正常/倒放/倾倒、密集反光和 4 类失败均可生成 |
| DATA-02 | 作为模型方，我要统一且无在线 GT 泄漏的 episode | C/F/A | D13 | schema、replay、split、data card | 50 条先验样本可读/回放；split 无泄漏；GT sidecar 隔离测试通过 |
| DATA-03 | 作为 D/E，我要从 canonical 数据得到 RLDS/LeRobot 输入 | C/D/E/F | D20 | 两转换器、manifest、反解测试 | 随机 10 step 反解物理动作误差 <1e-6；数据 tag 冻结 |
| AGT-01 | 作为用户，我要预设原始自然语言由 π0.5/Arm_A 理解，Supervisor 只按冻结 TaskProfile 管理阶段 | A/E | D17 | TaskEnvelope、TaskProfile、原始 instruction trace | 无 NLP Agent；无在线语义分派；无 pose/waypoint/grasp point；阶段与后置条件在赛前冻结 |
| AGT-02 | 作为系统，我要每个动作后重新观察、三帧 2 票核验并有界重决策 | A/B/F | D28 | FSM、单步滚动执行、Verifier、同角色恢复 trace | 新 observation 后才下第二步；当前固定角色重试 <=1；另一 VLA 不得接管 |
| AGT-03 | 作为工程团队，我要双 VLA 共享严格接口但固定控制各自机械臂 | A/D/E | D17 | health/infer/cancel schemas、角色白名单、适配器 | π0.5 只能发 Arm_A 动作；OpenVLA 只能发 Arm_B 动作；两模型不共享 norm stats |
| VLA-D1 | 作为 OpenVLA owner，我要复现官方基线并为 Arm_B 返回 canonical N×7 | D/A/B | D14 | 固定 commit、环境、服务、100 次压测 | health SHA 正确；相机预处理 checksum；仅在 `B_ONLY` 接收搬箱任务；P50/P95/显存和错误码齐全 |
| VLA-D2 | 作为 OpenVLA owner，我要给出工业微调的可复现实证 | D/C/F | D21-D33 | RLDS/OFT、checkpoint、base/tuned 表 | 同 seed/同协议；训练配置/数据/权重 SHA；改进和回退均报告 |
| VLA-E1 | 作为 π0.5 owner，我要复现 openpi 并为 Arm_A 返回 canonical N×7 | E/A/B | D15 | 固定 commit/submodule、LeRobot/norm、服务 | 独立 JAX 环境；真实 norm SHA；仅在 `A_ONLY` 执行装箱交接；cancel/超时/背压测试通过 |
| VLA-E2 | 作为 π0.5 owner，我要完成 Arm_A 装箱与交接的工业微调 | E/C/F | D21-D33 | LeRobot/openpi 微调 checkpoint、base/tuned 表 | 同 seed/同协议；训练配置/数据/权重/norm SHA；失败与同角色恢复均报告 |
| VER-01 | 作为评委，我要看到开放域感知的 mAP 和速度 | F/C | D31 | ID/OOD 集、全量 bbox 原始预测、mAP、P50/P95、失败样例 | split 冻结；原始预测可重算；YOLO 空检测/超时不阻塞 VLA；GT 仅离线 |
| VER-02 | 作为总 Agent，我要可靠区分成功、失败和不确定 | F/A | D26 | 3 帧 2 票、混淆矩阵、错误码 | 重复帧不重复投票；低置信不误判；GT 只做离线比对 |
| INT-01 | 作为团队，我要让 π0.5/Arm_A 完成装箱并把料箱送达交接区 | 全员/A | D18 | 20 局日志、视频、失败 Pareto | G3 通过；OpenVLA/Arm_B 固定搬箱合同同时完成 smoke，不得跨角色补位 |
| INT-02 | 作为队长，我要验证固定双 VLA 的安全交接生命周期 | A/B/D/E/F | D24 | 100 seed 交接对照、令牌时序、ADR | 三帧 2 票后才允许 `A_ONLY → HANDOFF_VERIFY → B_ONLY`；共享区始终仅一臂进入 |
| INT-03 | 作为评委，我要看到三任务族和失败恢复的稳定闭环 | 全员/F | D30 | 100 seed、恢复 trace、原始 JSONL | 三任务均从语言起跑；P0=0；失败只重试当前子任务 |
| INT-04 | 作为研究交付，我要可信的微调/核验/双模/随机化消融 | A/D/E/F | D33 | 消融表、95% CI、图表脚本 | 所有图表可从 raw 重生成；负结果保留 |
| REL-01 | 作为新成员，我要在干净机器按 README 复现 | F/B | D36 | 安装/启动录像、3 次一键启动 | 无作者缓存/绝对路径；结果格式和哈希一致 |
| REL-02 | 作为评委，我要完整技术报告和一镜到底演示 | F/A/全员 | D38 | 报告、成功/恢复视频、答辩题库 | `SCORE-*` 每一分有证据；限制不夸大；六项提交物齐全 |
| REL-03 | 作为队长，我要在截止前拥有可回退的最终包 | A/F/B | D40 | MANIFEST、哈希、上传回验、双备份 | 两机解包一致；无密钥/非法权重；9/3-5 只修提交阻断 |

## 3. Task 拆分规则

每个 Story 的 Issue 继续拆到如下颗粒度：

```text
Story: AGT-03 双执行器统一接口
├─ Task A（0.5 天）：冻结 health/infer/cancel Schema 与错误码
├─ Task D（1 天）：OpenVLA 服务字段映射 + 契约测试
├─ Task E（1 天）：openpi 服务字段映射 + 契约测试
├─ Task B（0.5 天）：7D 动作坐标/单位/夹爪往返测试
└─ Task F（0.5 天）：恶意元数据、超时、重复 ID 和回放证据
```

Task 必须有唯一 owner、明确路径、可复制命令、阈值、依赖和回退；不能以
“调研、推进、联调、优化”单独命名。
