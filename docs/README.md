# 文档索引

| 目录 | 内容 | 变更规则 |
|---|---|---|
| [`official/`](official/) | 两份官方 PDF 原件 | 不可编辑，SHA 校验 |
| [`assets/`](assets/) | 冻结架构图和分工图 | 不可静默改变边界 |
| [`source/`](source/) | 初版 40 天方案 DOCX | 可修订参考 |
| [`requirements/`](requirements/) | 官方需求、评分、提交物追踪 | A/F 维护 |
| [`project-management/`](project-management/) | 计划、每日任务、GitHub、看板、风险 | 按仓库证据更新 |
| [`architecture/`](architecture/) | 总 Agent 与跨进程接口合同 | Schema 优先，变更需评审 |

当前架构基线：四 Agent、无 NLP Agent、双机械臂、双 VLA 固定串行协作。
π0.5 固定控制 Arm_A 完成四零件装箱与静态中央交接，Supervisor 三帧核验后将
令牌从 `A_ONLY` 切到 `B_ONLY`，OpenVLA-OFT 固定控制 Arm_B 搬箱至
`FINISHED_01`。YOLO 是同步调用、失败非门控的评分 sidecar，独立保存
bbox/时延并离线计算 mAP，
GT 不进入在线 Agent。

优先入口：

- [仓库目录与文件规范](repository-structure.md)
- [官方需求不可变基线](requirements/official-requirements-baseline.md)
- [D01-D40 逐日计划](project-management/daily-plan.md)
- [GitHub 协作指南](project-management/github-collaboration-guide.md)
- [总 Agent 框架](architecture/agent-framework.md)
- [最终冻结场景与完整闭环](architecture/final-frozen-scene-and-flow.md)
- [接口契约](architecture/interface-contracts.md)
- [数据采集与其余五位成员执行指南](project-management/data-collection-and-five-member-execution-guide.md)
- [当前 YOLO 评分旁路决策 ADR-0003](architecture/ADR-0003-yolo-scoring-sidecar.md)
- [已废止的 YOLO 硬门控 ADR-0002](architecture/ADR-0002-four-agent-yolo-gate.md)
