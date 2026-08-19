# 可修订方案源文件

`XH-202607_initial_plan_v1.0.docx` 是 2026-07-19 形成的六人 40 天初版方案，
SHA-256 为
`4360A1D56F3A48DA83680FF63C15D06FB6F9D893E76EF25A6249493990FC60AB`。

它是可修订的规划输入，不是官方要求。仓库中的 `v1.0` 文件是原始历史快照，
不得原地覆写；若确需形成新版，应新增 `v1.1` 或更高版本，并在 PR 中说明差异。
实际执行以版本化的 Markdown 项目管理文档、当天仓库证据和 Gate 结果为准；
与官方 PDF 冲突时必须服从官方 PDF，与冻结架构图冲突时必须服从冻结图。

运行 `python scripts/verify_project_frozen_inputs.py` 可校验该 v1.0 快照和两张
团队冻结图。该校验不代表 DOCX 具有官方文件地位。

`source-code-parameter-and-documentation-guide.docx` 保留为 2026-08-07 的源码手册
基线；`source-code-parameter-and-documentation-guide-v2.docx` 是面向当前仓库的
可修订版本，已补充 V2 人工工业场景、采集入口与 V1/V2 适用边界。两者都不是
运行时合同真源；场景参数以 `simulation/configs/single_bin_scene_v2.json` 和
`simulation/v2_scene_contract.py` 为准。
