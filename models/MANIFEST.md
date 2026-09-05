# Model Artifact Manifest

本文件只登记外部模型制品，不保存权重。D/E 每产生一个可评测 checkpoint，就追加
一行；F 在 G3–G7 前复核 URI 可访问、SHA-256 匹配且许可证/来源明确。

| Artifact ID | Model | Purpose | Upstream commit | External URI | Checkpoint SHA-256 | Norm stats SHA-256 | License/source | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | `NOT_AVAILABLE` |
| `yolo_manual1394_wrench_yolo11n_e5_cpu` | YOLO11n | Wrench-recall-focused V2 perception | Pending merge | [model artifact repository](https://github.com/RUIJIAN-HUANG/industrial-agent-vla-model-yolo-manual800/tree/codex/yolo-manual1394-wrench-artifact/manual1394_wrench) | `sha256:6bb9d5006e732426458322e7258d3043e367317dfd46ae54920f9605a90b9536` | N/A | Human-reviewed internal simulator data | F | `FROZEN` |

规则：

- URI 不得包含 token、临时签名参数或个人绝对路径；
- 禁止使用 `latest`、文件夹名或昵称代替摘要；
- 状态只能使用 `NOT_AVAILABLE`、`TRAINING`、`CANDIDATE`、`FROZEN`、`RETIRED`；
- `FROZEN` 制品必须关联实验记录、评测报告和可复现环境。
