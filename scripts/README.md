# Scripts

| 脚本 | 用途 |
|---|---|
| `verify_official_baselines.py` | 只校验唯二官方 PDF 的 SHA-256 |
| `verify_project_frozen_inputs.py` | 校验两张团队冻结图和初版 DOCX 快照 |
| `run_mock_demo.py` | 运行成功、同策略恢复、执行器切换三个总 Agent Mock 场景 |
| `check_repository_hygiene.py` | 拒绝误提交的权重、数据录包、视频、密钥、缓存和超大文件 |

```powershell
python scripts/check_repository_hygiene.py
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python scripts/run_mock_demo.py
```

Mock 不加载真实 OpenVLA-OFT、π0.5 或仿真平台。
