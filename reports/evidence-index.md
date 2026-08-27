# Competition Evidence Index

本索引连接官方要求、实验、日志、视频和最终报告。当前尚未产生真实仿真/VLA
证据；接口测试或 Mock 结果不得登记为真实模型成功。

| Evidence ID | Req/Gate | Description | Git commit/config | External URI | SHA-256 | Reproduction command | Owner | Status |
|---|---|---|---|---|---|---|---|---|
| EVID-000 | G0 | 仿真平台 1000 步、相机落盘、三次重启 | — | — | — | — | B/F | `PENDING` |
| EVID-001 | G3 | 首个真实 VLA 20 局闭环 | — | — | — | — | D/E/F | `PENDING` |
| EVID-002 | G5 | 三任务族与失败恢复 100 局 | — | — | — | — | A–F | `PENDING` |
| EVID-003 | DEL-04 | 自然语言到执行的完整仿真视频 | — | — | — | — | A/F | `PENDING` |
| EVID-YOLO-MANUAL800 | YOLO | Manual-800 YOLO checkpoint identity, class map, service config, and three-camera probe procedure | this PR | external artifact required; see `models/MODEL_CARD_yolo_manual800.md` | `sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f` | `python scripts/run_yolo_three_camera_probe.py --observation-json <validated-observation.json> --output-json reports/yolo/manual800/three-camera-probe-summary.json --evidence-jsonl artifacts/detection/manual800-three-camera-probe.jsonl` | F | `PARTIAL` |

状态只能使用 `PENDING`、`PARTIAL`、`REPRODUCIBLE`、`REJECTED`。登记外部证据时
必须提供不可变 SHA-256；禁止记录含访问 token 的临时下载链接。
