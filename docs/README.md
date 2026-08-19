# 文档索引

| 目录 | 内容 | 变更规则 |
|---|---|---|
| [`official/`](official/) | 两份官方 PDF 原件 | 不可编辑，SHA 校验 |
| [`assets/`](assets/) | 冻结架构图和分工图 | 不可静默改变边界 |
| [`source/`](source/) | 初版 40 天方案 DOCX | 可修订参考 |
| [`requirements/`](requirements/) | 官方需求、评分、提交物追踪 | A/F 维护 |
| [`project-management/`](project-management/) | 计划、每日任务、GitHub、看板、风险 | 按仓库证据更新 |
| [`architecture/`](architecture/) | 总 Agent 与跨进程接口合同 | Schema 优先，变更需评审 |

当前文档采用双层口径：

- **当前工业场景（V2）**：双 Franka、三固定相机、8 个程序化工业零件、
  `2×4` 料箱与中央提梁，用于人工键盘 Canonical Episode 采集；
- **自动闭环兼容基线（V1）**：四 Agent、无 NLP Agent、双 VLA 固定串行协作，
  π0.5/Arm_A 处理 P01-P04，Supervisor 三帧核验，OpenVLA-OFT/Arm_B 搬箱。

两条链路共享 7D 动作、CAS、三相机 ID、GT 隔离和双臂令牌安全边界，但 V2 尚未
替换 `single_bin_pack_handoff_v1` 的冻结指令与 Supervisor 后置条件。YOLO 仍是
失败非门控的评分 sidecar。

优先入口：

- [仓库目录与文件规范](repository-structure.md)
- [官方需求不可变基线](requirements/official-requirements-baseline.md)
- [D01-D40 逐日计划](project-management/daily-plan.md)
- [GitHub 协作指南](project-management/github-collaboration-guide.md)
- [总 Agent 框架](architecture/agent-framework.md)
- [V2 人工工业采集说明](v2-manual-industrial-collection.md)
- [场景与自动闭环边界](architecture/final-frozen-scene-and-flow.md)
- [接口契约](architecture/interface-contracts.md)
- [数据采集与其余五位成员执行指南](project-management/data-collection-and-five-member-execution-guide.md)
- [当前 YOLO 评分旁路决策 ADR-0003](architecture/ADR-0003-yolo-scoring-sidecar.md)
- [已废止的 YOLO 硬门控 ADR-0002](architecture/ADR-0002-four-agent-yolo-gate.md)
