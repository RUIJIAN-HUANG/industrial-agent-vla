# Scripts

| 脚本 | 用途 |
|---|---|
| `verify_official_baselines.py` | 校验两份官方 PDF、两张冻结图和初版 DOCX 的 SHA-256 |
| `run_mock_demo.py` | 运行成功、同策略恢复、执行器切换三个总 Agent Mock 场景 |

```powershell
python scripts/verify_official_baselines.py
python scripts/run_mock_demo.py
```

Mock 不加载真实 OpenVLA-OFT、π0.5 或仿真平台。
