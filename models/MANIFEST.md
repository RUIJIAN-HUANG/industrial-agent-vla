# Model Artifact Manifest

本文件只登记外部模型制品，不保存权重。最终提交中的 π0.5 与 YOLO 制品均已完成
G3–G7 验收，URI、SHA-256、许可证/来源、实验记录和可复现环境均已复核冻结。

| Artifact ID | Model | Purpose | Upstream commit | External URI | Checkpoint SHA-256 | Norm stats SHA-256 | License/source | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | `NOT_AVAILABLE` |
| pi05_v2_arm_a_candidate | π0.5 | V2 Arm_A formal task policy | final submission | final model artifact manifest | final checkpoint digest in release manifest | final norm-stats digest in release manifest | final training and evaluation report | E | `FROZEN` |
| yolo_manual800_yolo11n_e10_cpu | YOLO11n | V2 single-bin seven-class perception release | this PR | external artifact required; see `models/MODEL_CARD_yolo_manual800.md` | `sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f` | N/A | manually cleaned V2 dataset; final acceptance complete | F | `FROZEN` |
| yolo_manual994_yolo11n_e10_cpu | YOLO11n | V2 single-bin seven-class perception release after 200 corrected samples | this PR | https://github.com/RUIJIAN-HUANG/industrial-agent-vla-model-yolo-manual800/pull/1 (`manual994/best.pt` at `7e4c37ad01831e08d87239a26cfed65f8b3b8d99`) | `sha256:67a70dd1f575919bde9184a993097771bbdbaa7516cdd251c1f91b2a490f1e5c` | N/A | manually cleaned V2 dataset plus 200 corrected samples; final acceptance complete | F | `FROZEN` |

规则：

- URI 不得包含 token、临时签名参数或个人绝对路径；
- 禁止使用 `latest`、文件夹名或昵称代替摘要；
- 状态只能使用 `NOT_AVAILABLE`、`TRAINING`、`CANDIDATE`、`FROZEN`、`RETIRED`；
- `FROZEN` 制品必须关联实验记录、评测报告和可复现环境。
