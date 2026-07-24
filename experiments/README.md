# Experiments

负责人：实验执行人，汇总核验：F。本目录保存能复现实验的**定义和摘要**，原始日志、
缓存和 tracker 目录不进入 Git。

每个实验使用 `YYYYMMDD_<model>_<task>_<short-id>/`，至少包含：

- `README.md`：假设、负责人、日期、关联 Issue/Gate；
- 冻结配置或配置路径；
- Git Commit、数据 Manifest SHA、checkpoint/norm stats SHA；
- seed 列表、环境版本、执行命令；
- 指标摘要、失败码分布和外部证据链接；
- 结论、限制以及是否影响 G3/G4 路由决策。

禁止只上传截图而没有配置和原始证据索引，也禁止提交 `runs/`、W&B/MLflow 缓存或
完整 JSONL 日志。
