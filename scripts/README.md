# Scripts

| 脚本 | 用途 |
|---|---|
| `verify_official_baselines.py` | 只校验唯二官方 PDF 的 SHA-256 |
| `verify_project_frozen_inputs.py` | 校验两张团队冻结图和初版 DOCX 快照 |
| `run_mock_demo.py` | 运行正常闭环、Arm_A 恢复、Arm_B 恢复三个固定双 VLA 串行 Mock |
| `evaluate_detection_map.py` | 离线计算 YOLO bbox AP50/AP75/mAP50:95、P/R 与 P50/P95 时延，并保留原始预测 |
| `check_repository_hygiene.py` | 拒绝误提交的权重、数据录包、视频、密钥、缓存和超大文件 |

```powershell
python scripts/check_repository_hygiene.py
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python scripts/run_mock_demo.py
```

Mock 不加载真实 OpenVLA-OFT、π0.5 或仿真平台；其中 `durable_ack=true` 只模拟
“持久化成功后再授权”的顺序，不证明真实文件系统 fsync。每个场景
验证 `π0.5 → 候选预检 → 锁臂后三帧 2/3 交接核验 → OpenVLA-OFT` 的调用顺序、
`A_ONLY → HANDOFF_VERIFY → B_ONLY` 的令牌顺序。候选预检事件
`handoff.candidate_checked` 在进入交接的运行中至少出现一次，并可按重试需要出现
1..N 次；不可逆里程碑只校验
`handoff.verified → handoff.ready` 的事件顺序。候选帧不进入三帧投票，只有
`handoff.ready` 表示 Arm_B 可以执行。

YOLO 评测是同步调用、失败非门控的评分 sidecar：在线检测空结果、超时或坏响应必须留证，但不得
阻止 π0.5/OpenVLA-OFT 主控制链路。以下离线命令是 GT 与预测唯一允许汇合的位置。

检测评测必须使用两个物理隔离的输入：在线 YOLO 导出的原始预测，以及只允许
离线评测进程读取的冻结 COCO GT：

```powershell
python scripts/evaluate_detection_map.py `
  --ground-truth <frozen-coco-annotations.json> `
  --predictions <online-raw-predictions.json> `
  --output-dir <evidence-directory> `
  --engine pycocotools `
  --require-trace-linkage
```

输出保留输入预测的逐字节副本 `raw_predictions.json`，以及包含输入哈希、
AP50/AP75/mAP50:95、各类别 AP、IoU=0.50 下 aggregate/per-class
Precision/Recall、P50/P95 检测时延的 `detection_metrics.json`；不会复制 GT。

`--engine minimal` 是无第三方依赖的 bbox-only、area=all、maxDet=100、
101-point 插值回退，不支持 COCO `iscrowd/ignore` 语义；适合 CI/早期基线，
但不是官方唯一口径。最终比赛证据优先使用 `pycocotools`。若未安装却明确选择
该引擎，脚本会以非零状态和清晰错误退出，不会自动伪造 COCOeval 分数；只有
`--engine auto` 会显式记录回退警告，且 crowd/ignore 数据禁止不安全回退。
