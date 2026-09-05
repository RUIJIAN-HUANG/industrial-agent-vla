# 文档索引

| 目录 | 内容 | 变更规则 |
|---|---|---|
| [`official/`](official/) | 两份官方 PDF 原件 | 不可编辑，SHA 校验 |
| [`assets/`](assets/) | 冻结架构图和分工图 | 不可静默改变边界 |
| [`source/`](source/) | 初版 40 天方案 DOCX | 可修订参考 |
| [`requirements/`](requirements/) | 官方需求、评分、提交物追踪 | A/F 维护 |
| [`official-test-guide.md`](official-test-guide.md) | 官方测试、Isaac Sim 验收、双模型提交包检查 | F 维护 |
| [`project-management/`](project-management/) | 看板、每日任务和自动发布规则 | 按仓库证据更新 |
| [`architecture/`](architecture/) | 总 Agent 与跨进程接口合同 | Schema 优先，变更需评审 |

当前文档采用单一正式口径：

- **正式工业场景与闭环（V2）**：V2 Canonical Episode、总控、YOLO、单一 π0.5
  双臂路由、V2 在线观测和 Isaac 连续闭环。

优先入口：

- [仓库目录与文件规范](repository-structure.md)
- [官方需求不可变基线](requirements/official-requirements-baseline.md)
- [当前项目看板](project-management/dashboard.md)
- [总 Agent 框架](architecture/agent-framework.md)
- [V2 人工工业采集说明](v2-manual-industrial-collection.md)
- [场景与自动闭环边界](architecture/final-frozen-scene-and-flow.md)
- [接口契约](architecture/interface-contracts.md)
- [当前 YOLO 评分旁路决策 ADR-0003](architecture/ADR-0003-yolo-scoring-sidecar.md)
