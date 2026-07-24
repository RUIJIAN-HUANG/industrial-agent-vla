# XH-202607 每日任务公告模板

> 日期：`YYYY-MM-DD` / `Dxx`
> 发布：A（09:00 前）
> 集成门：18:00
> 当日 Gate：`无 / Gx`
> 状态来源：GitHub Issue、PR、CI、日志和可复现文件；口头进度不作为完成证据。

## 1. 今日唯一目标

用一句可验收的话描述今日全队唯一目标：

> 在 ______ 环境中，以 ______ 配置完成 ______，并由 ______ 证据证明。

## 2. 昨日事实与偏差

| 项目 | 计划 | 实际证据 | 偏差 | 今日处理 |
|---|---|---|---|---|
| Gate/主目标 |  | PR/日志/报告链接 | `0 / +N 天` |  |
| P0 阻塞 |  | Issue 链接 |  |  |

## 3. A-F 当日任务

| 角色 | P | 今日唯一主任务 | 输入/依赖 | 必交文件或证据 | Definition of Done | 截止 | 状态 |
|---|---|---|---|---|---|---|---|
| A | P0 |  |  |  |  | 17:00 | Todo |
| B | P0 |  |  |  |  | 17:00 | Todo |
| C | P0 |  |  |  |  | 17:00 | Todo |
| D | P0 |  |  |  |  | 17:00 | Todo |
| E | P0 |  |  |  |  | 17:00 | Todo |
| F | P0 |  |  |  |  | 17:00 | Todo |

### DoD 强制项

- [ ] 交付已在个人分支提交，并关联 Issue；
- [ ] 已创建或更新 Draft PR，PR 描述写明测试方法；
- [ ] CI 通过，或明确记录与本 PR 无关的既有失败；
- [ ] 配置、seed、数据/模型/代码 SHA 和运行环境可追溯；
- [ ] 原始日志、失败样例和结果文件已按约定保存；
- [ ] 文档与实际接口一致；
- [ ] 不含密钥、仿真 GT 泄漏、大模型权重或未授权数据；
- [ ] 需要他人消费的接口已由至少一名消费方验收。

## 4. 依赖与交接

| From | To | 交接物 | 最晚时间 | 验收人 | 未交付回退 |
|---|---|---|---|---|---|
|  |  |  | 14:00 |  |  |

## 5. 18:00 集成门

统一执行：

```powershell
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
git diff --check
```

对 GPU、仿真和模型任务，Issue 还必须附启动命令、硬件、P50/P95、显存峰值、
成功/失败局数及至少一个失败样例。

## 6. 风险与升级

| 风险/阻塞 | 概率 | 影响 | Owner | 触发时间 | 今日措施 | 需 A 决策 |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  | 是/否 |

- P0：立即通知 A、F，不等待晚会；当天给出继续/降级/停止决定。
- P1：14:00 前暴露，18:00 集成门决定次日资源。
- 连续两天未达到 DoD：拆小任务或执行 Gate 中的降级方案，禁止只顺延日期。

## 7. 日终记录

| 角色 | 状态 | PR/证据 | 未完成原因 | 次日第一动作 |
|---|---|---|---|---|
| A | Done/Partial/Blocked |  |  |  |
| B | Done/Partial/Blocked |  |  |  |
| C | Done/Partial/Blocked |  |  |  |
| D | Done/Partial/Blocked |  |  |  |
| E | Done/Partial/Blocked |  |  |  |
| F | Done/Partial/Blocked |  |  |  |

A 在日终只更新事实、偏差和决策；不得把 `Partial` 写成 `Done`。
