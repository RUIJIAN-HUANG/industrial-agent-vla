# Models

负责人：D/E，复现核验：F。本目录只保存模型卡、来源、许可证、下载方法、兼容性和
固定摘要；真实服务代码进入 `services/`，模型权重进入外部制品存储。

每个模型版本至少记录：

- 上游仓库 URL 与 Commit；
- base/fine-tuned checkpoint SHA-256；
- norm stats SHA-256；
- 训练数据 Manifest、任务范围和相机顺序；
- 输入/输出合同版本、依赖环境和已知限制；
- 复现命令及评测报告链接。

禁止提交 `.ckpt/.pt/.pth/.safetensors/.onnx/.engine` 等权重或导出引擎。推荐文件名：
`MODEL_CARD_<model>_<version>.md`、`CHECKSUMS_<model>_<version>.json`。
