# 官方不可变基线

本目录保存赛题 `XH-202607` 的两份官方文件。它们是项目需求、验收与
冲突裁决的最高优先级依据；文件必须保持原字节，不得直接编辑或重新导出。

| 文件 | SHA-256 |
|---|---|
| `XH-202607_competition_spec.pdf` | `FDC21B1C0EDAA48BD2CDE22E5B103F458F5106759ACD4D9C65236549D4695D25` |
| `XH-202607_official_QA.pdf` | `0A381757E35EE402E954CCB34CA0A5453DE4119AABEED1165AFD66666FC05731` |

校验示例：

```powershell
python scripts/verify_official_baselines.py
```

若校验值不一致，立即停止需求变更和正式实验，由项目负责人 A 恢复基线。
逐条需求追踪见
[`../requirements/official-requirements-baseline.md`](../requirements/official-requirements-baseline.md)。
