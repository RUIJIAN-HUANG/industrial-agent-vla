# XH-202607 官方需求不可变基线

> 状态：`FROZEN`
> 基线日期：2026-07-24
> 负责人：A
> 规则：本文件是两份官方 PDF 的工程化索引，不替代原文。发生歧义时必须
> 回到对应页原文；任何团队文档、代码或口头约定均不得降低官方要求。

## 1. 唯二官方真源

| ID | 仓库文件 | 页数 | SHA-256 |
|---|---|---:|---|
| SRC-01 | [`../official/XH-202607_competition_spec.pdf`](../official/XH-202607_competition_spec.pdf) | 9 | `FDC21B1C0EDAA48BD2CDE22E5B103F458F5106759ACD4D9C65236549D4695D25` |
| SRC-02 | [`../official/XH-202607_official_QA.pdf`](../official/XH-202607_official_QA.pdf) | 4 | `0A381757E35EE402E954CCB34CA0A5453DE4119AABEED1165AFD66666FC05731` |

运行 `python scripts/verify_official_baselines.py` 校验原字节。校验失败时停止
需求变更、正式训练和评测，按 P0 处理。

## 2. 冲突裁决顺序

1. 官方答疑对官方比赛方案的补充/澄清；
2. 官方比赛方案；
3. 当前冻结架构图 `docs/architecture/assets/four-agent-fixed-dual-vla-architecture-v4-zh.png`；
4. 冻结分工图 `docs/assets/team-roles-frozen.png`；
5. 原始架构快照 `docs/assets/system-architecture-frozen.png`；
6. 本追踪矩阵与版本化接口文档；
7. 40 天计划、Issue、会议决定；
8. 初版 DOCX，仅作为可修订参考。

低优先级材料与高优先级材料冲突时，不得“折中解释”，必须修改低优先级
材料。任何人发现冲突应创建 P0/P1 Issue，并通知 A 与 F。

## 3. 硬性能力追踪矩阵

| Req ID | 官方依据 | 不可弱化的要求 | 冻结架构归属 | 必须形成的证据 | Owner |
|---|---|---|---|---|---|
| REQ-SYS-001 | SRC-01 p1-p2；p3 | 系统覆盖环境物体感知、自然语言理解、任务序列分解、决策与执行的完整链路 | 总 Agent + VLA 执行 Agent + 仿真/机械臂 + 环境反馈 | 一镜到底日志/视频；状态迁移 trace；固定 seed 成功率 | A/F |
| REQ-DATA-001 | SRC-01 p1；SRC-02 p1 Q1-Q2 | 构建工业工具/零件场景数据；标注类别、位置等；公开/仿真/生成式数据均可，但必须能评价工业域泛化 | 场景/资产/数据链路 | DATA_CARD、schema、split manifest、许可、生成脚本、样例审计 | C/F |
| REQ-PER-001 | SRC-01 p2；SRC-02 p1 Q2 | 在密集、堆叠、杂乱等工业场景具备开放域感知、识别和定位能力，并评价微调后泛化 | 环境反馈/感知验证 | mAP、推理延迟、ID/OOD 分项、base vs fine-tuned | F/C |
| REQ-NLP-001 | SRC-01 p2 | 理解简单工业自然语言指令，提取目标工具/零件与动作要求 | π0.5/Arm_A 接收预设原始语言；本基线不设置 NLP Agent，Supervisor 只封装/透传 | 冻结语言集、原始指令 trace、拒错/澄清测试 | A/E |
| REQ-PLAN-001 | SRC-01 p2；SRC-02 p3 Q8 | 将高层指令拆为符合机器人作业逻辑的有序任务序列 | π0.5 的语言/动作序列能力 + 赛前冻结 TaskProfile/FSM；交接后 OpenVLA/Arm_B 执行固定搬箱阶段 | 任务阶段、依赖/后置条件、动作序列与推进/同角色恢复 trace | A/D/E |
| REQ-TASK-001 | SRC-01 p3 | 对散乱零件/工具识别、抓取，并放入指定多格料箱的正确格子 | VLA 执行 + 仿真环境 + 核验 | 指定格任务连续评测与失败视频 | B/C/D/E/F |
| REQ-FAIL-001 | SRC-01 p3；SRC-02 p1 Q3-Q4 | 每个执行步骤后重新感知并比对预期；摆放失败时自主识别、重决策并生成修正序列，不得简单终止 | Supervisor + 环境反馈/三帧 2 票核验 + 当前固定 VLA 同角色重规划 | 至少抓空、错格、倾倒/倒放、掉落故障注入；恢复率与 trace | A/B/D/E/F |
| REQ-MEM-001 | SRC-02 p1 Q3 | 决策模块具备记忆、闭环反思、重决策、智能交互能力 | 总 Agent 状态/记忆/结构化 Reflection | episode 事件日志、预期/实际差异、重试上限测试 | A/F |
| REQ-MA-001 | SRC-02 p1 Q3；p3 Q8 | 体现多智能体协同；官方示例为多个执行机构的并/串行分工 | 四 Agent 固定协作：Supervisor、π0.5/Arm_A、OpenVLA/Arm_B、YOLO；双臂以令牌串行进入共享区 | 完整事件链、交接令牌与双臂互斥 trace；一镜到底协作录像 | A/B/D/E/F |
| REQ-SIM-001 | SRC-02 p1 Q5；p2 Q6 | 初赛必须在仿真环境实现；推荐 Isaac Sim 或 Gazebo | 项目工程实现固定采用 Isaac Sim + 双 Franka | 一键启动、场景包、headless 运行、固定 seed 回归 | B/F |
| REQ-SCENE-001 | SRC-02 p2-p3 Q8 | 重点工业示例包含密集、高反光圆柱件，以及正常、倒放、倾倒状态 | 场景/资产/数据 + 核验 | 场景参数、状态标签、随机化范围、各状态评测 | C/F |
| REQ-ADV-001 | SRC-02 p3 Q8 | 支持“最多区域装箱”、倒放转正常、指定行列格；料箱未满时先装满，再搬运/叠放 | 冻结 TaskProfile/FSM + VLA 语言理解与执行 + 核验 | 三类冻结任务族、分项成功率、长时序 trace | A/B/C/D/E/F |
| REQ-FT-001 | SRC-01 p3；SRC-02 p1 Q2 | 可调用现有大模型+prompt；工业场景微调是明确加分方向，必须说明改进效果 | 项目冻结为 OpenVLA-OFT 与 π0.5 都做工业微调；感知模型独立评测 | 两个 VLA 各自 base/fine-tuned 同协议对照、配置与 checkpoint SHA | D/E/F |
| REQ-HW-001 | SRC-01 p4-p5；SRC-02 p1 Q5、p2 Q7 | 真机不是初赛仿真硬门槛，但真实机械臂+相机案例单列 10 分 | 仿真环境执行接口可迁移；真机为条件增强 | 有真机则视频/配置；无真机不得把接口冒充验证 | A/B/F |

### 边界解释

- 本基线不设置 NLP Agent。Supervisor 不做在线自然语言理解，不判断任务复杂度，
  只加载赛前冻结 TaskProfile 并把预设原始指令透传给 π0.5/Arm_A。TaskProfile
  只描述阶段、依赖和可观察后置条件，不得包含目标坐标、轨迹点或抓取姿态。
- 四 Agent 与双臂职责冻结：π0.5 只控制 Arm_A 完成装箱并把料箱放到
  `HANDOFF_CENTER` 后退回 `HOME_A`；OpenVLA 只在 `B_ONLY` 阶段控制 Arm_B
  把同一料箱搬到 `FINISHED_01` 后退回 `HOME_B`。任一故障都不得跨角色接管。
- Supervisor 只按 `A_ONLY → HANDOFF_VERIFY → B_ONLY` 管理生命周期。交接条件
  必须由三个去重帧中的至少两票成立；任一时刻只允许一条机械臂进入共享区。
- YOLO Agent 是同步调用、失败非门控的评分 sidecar：保存 bbox、类别、置信度、
  ROI 计数、占用状态与时延，
  为在线核验提供非真值证据，并在离线冻结 GT 上计算 AP50、AP75、mAP50:95。
  YOLO 空检测、超时或坏包不得阻断 VLA，也不得改变两个 VLA 的固定职责。
- GT、仿真真值、人工框、目标坐标和抓取点不得回灌总控、YOLO、任一 VLA 或
  在线核验器；GT 只与原始预测在离线 Evaluator 汇合。
- OpenVLA-OFT 与 π0.5 都必须完成各自固定角色的工业场景微调和 base/tuned
  对照；不得根据复杂度、置信度或失败原因互换机械臂、任务阶段或恢复职责。

## 4. 评分基线（100 分）

| Score ID | 分值 | 官方评分点 | 项目最低证据 | Owner |
|---|---:|---|---|---|
| SCORE-01 | 20 | 完整覆盖“感知-决策-执行” | 四 Agent 完整 trace、双臂交接视频、状态图、连续回合 | A/F |
| SCORE-02 | 15 | 视觉识别、任务分解、Agent 框架创新与迁移复用 | 统一契约、适配器、GT 隔离、恢复消融、新布局迁移 | A/C/F |
| SCORE-03 | 10 | 感知 mAP 与推理速度 | 冻结 val/test、mAP、P50/P95 延迟、硬件说明 | F/C |
| SCORE-04 | 10 | 任务序列合理性与执行成功率 | TaskProfile/FSM、固定双 VLA 动作序列、三任务族成功率、失败码分布 | A/D/E/F |
| SCORE-05 | 15 | 工业场景模型微调及改进效果 | base vs tuned、同 seed/配置、置信区间、负结果 | D/E/F |
| SCORE-06 | 5 | 代码清晰、注释完整、易复现 | CI、测试、README、版本锁、第二人复现 | F/全员 |
| SCORE-07 | 5 | 仿真验证且可扩展至真实环境 | 场景包、适配器边界、仿真复现录像 | B/F |
| SCORE-08 | 10 | 真实机械臂和相机验证案例 | 真机完整案例；无设备则诚实记为未覆盖 | A/B/F |
| SCORE-09 | 5 | 技术报告逻辑和数据 | 需求-实验-结论证据索引、原始结果可重生成 | A/F |
| SCORE-10 | 5 | 演示视频直观展示全过程 | 一镜到底 + 失败恢复 + 指标 overlay | F/A |

## 5. 官方提交物追踪

| Deliverable ID | 官方依据 | 必须包含 | 仓库/交付位置 | 验收 |
|---|---|---|---|---|
| DEL-01 | SRC-02 p3 Q9.1 | 感知、决策、执行最终推理代码与训练/微调模型；模块可独立运行并输出结果 | `src/`、`services/`、`scripts/`、`models/MANIFEST.md` | 干净环境 smoke |
| DEL-02 | SRC-02 p3 Q9.2 | 完整训练代码、预训练权重来源、工业数据/生成式数据说明 | `services/openvla_oft/`、`services/pi05/`、`data/` 清单 | 小样本训练 smoke |
| DEL-03 | SRC-02 p3 Q9.3 | Docker/打包环境、场景与仿真 Python 必要文件 | `simulation/`、容器与版本清单 | headless 固定 seed |
| DEL-04 | SRC-01 p4；SRC-02 p4 Q9.4 | 从自然语言输入到任务序列/执行的仿真验证视频；真机视频可附 | 外部视频清单 + `reports/evidence-index.md` | 完整播放和哈希 |
| DEL-05 | SRC-01 p4；SRC-02 p4 Q9.5 | 总体方案、模块设计、创新、测试；重点说明开放域感知、决策、微调和 Agent | `reports/technical-report.*` | 评分项反查 |
| DEL-06 | SRC-01 p4；SRC-02 p4 Q9.6 | 各模块环境、依赖、运行步骤、硬件、通信接口和集成方式 | `README.md`、`docs/architecture/` | 第二人复现 |

正式提交截止为 **2026-09-05 前**，方式和压缩包命名见 SRC-01 p5-p7。
项目内部 D40 为 2026-09-02，9 月 3-5 日只允许复现、校验、上传和应急回退。

## 6. 每周需求审计

每周由 A 与 F 完成：

1. 运行基线哈希校验；
2. 对每个 `REQ-*` 填写最新 PR、测试、日志或视频证据；
3. 对 `SCORE-*` 标记 `No evidence / Partial / Reproducible`；
4. 对任何“只有接口、没有运行证据”的项保持未完成；
5. 将缺口转成有 owner、日期和 DoD 的 Issue；
6. 若范围压缩，先保护 REQ-SYS、REQ-PLAN、REQ-FAIL、REQ-SIM 与 DEL-01~06。
