# Data

负责人：C，QA/备份：F。本目录只保存**数据元信息和可公开的小型测试夹具**，不保存
训练集、仿真批量轨迹或相机录像。

允许提交：

- `DATA_CARD*.md`：来源、许可证、采集方法、字段、质量和已知偏差；
- `MANIFEST*.md`：外部制品 URI、SHA-256、样本数、split 和生成 Commit；
- `schema*.json`：canonical/RLDS/LeRobot 字段定义；
- `fixtures/`：单元测试所需、去敏且体积很小的合成样本。

不得提交：`raw/`、`processed/`、`generated/`、`.npy/.npz/.h5`、数据缓存或含
ground-truth 的在线推理输入。正式数据版本应以不可变清单和哈希引用外部制品。

推荐命名：`DATA_CARD_<dataset>_<version>.md`、
`MANIFEST_<dataset>_<version>.md`。
