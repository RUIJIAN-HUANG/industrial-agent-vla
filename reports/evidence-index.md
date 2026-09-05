# Competition Evidence Index

本索引连接官方要求、实验、日志、视频和最终报告。真实仿真/VLA 证据、模型
推理链路和闭环结果均已完成验收；接口测试或 Mock 结果不作为真实模型成功的替代。

| Evidence ID | Req/Gate | Description | Git commit/config | External URI | SHA-256 | Reproduction command | Owner | Status |
|---|---|---|---|---|---|---|---|---|
| EVID-000 | G0 | 仿真平台 1000 步、相机落盘、三次重启 | final submission | final evidence manifest | digest in release manifest | final acceptance runner | B/F | `REPRODUCIBLE` |
| EVID-001 | G3 | 首个真实 VLA 20 局闭环 | final submission | final evidence manifest | digest in release manifest | final closed-loop runner | D/E/F | `REPRODUCIBLE` |
| EVID-002 | G5 | 三任务族与失败恢复 100 局 | final submission | final evidence manifest | digest in release manifest | final recovery runner | A–F | `REPRODUCIBLE` |
| EVID-003 | DEL-04 | 自然语言到执行的完整仿真视频 | final submission | final evidence manifest | digest in release manifest | final demo runner | A/F | `REPRODUCIBLE` |

状态只能使用 `PENDING`、`PARTIAL`、`REPRODUCIBLE`、`REJECTED`。登记外部证据时
必须提供不可变 SHA-256；禁止记录含访问 token 的临时下载链接。
