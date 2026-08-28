# Model Artifact Manifest

本文件只登记外部模型制品，不保存权重。D/E 每产生一个可评测 checkpoint，就追加
一行；F 在 G3–G7 前复核 URI 可访问、SHA-256 匹配且许可证/来源明确。

| Artifact ID | Model | Purpose | Upstream commit | External URI | Checkpoint SHA-256 | Norm stats SHA-256 | License/source | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | `NOT_AVAILABLE` |
| yolo_manual800_yolo11n_e10_cpu | YOLO11n | V2 single-bin seven-class perception candidate | this PR | external artifact required; see `models/MODEL_CARD_yolo_manual800.md` | `sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f` | N/A | manually cleaned V2 dataset | F | `CANDIDATE` |
| yolo_manual994_yolo11n_e10_cpu | YOLO11n | V2 single-bin seven-class perception candidate after 200 corrected samples | this PR | external artifact required; see `models/MODEL_CARD_yolo_manual994.md` | `sha256:67a70dd1f575919bde9184a993097771bbdbaa7516cdd251c1f91b2a490f1e5c` | N/A | manually cleaned V2 dataset plus 200 corrected samples | F | `CANDIDATE` |

规则：

- URI 不得包含 token、临时签名参数或个人绝对路径；
- 禁止使用 `latest`、文件夹名或昵称代替摘要；
- 状态只能使用 `NOT_AVAILABLE`、`TRAINING`、`CANDIDATE`、`FROZEN`、`RETIRED`；
- `FROZEN` 制品必须关联实验记录、评测报告和可复现环境。
