# 贡献与协作规范

本仓库由 6 名成员在短周期内并行开发。所有贡献都应当可追踪、可复现、可评审、可回退。完整的新手操作步骤见
项目状态与当前工作入口见 [项目看板](docs/project-management/dashboard.md)。

## 1. 不可变更的项目基线

1. `docs/official/` 下两份比赛官方 PDF 是需求与验收的唯一最高优先级依据，不得改写、替换或用团队推测覆盖。
2. 当前系统架构是双 Franka、固定串行、单 π0.5 双臂服务：
   π0.5/Arm_A → `HANDOFF_VERIFY` → π0.5/Arm_B，YOLO 为同步调用、失败非门控的
   评分 sidecar。Arm_B 的控制权仍必须经过 `B_ONLY` 令牌；不得因复用模型服务而
   绕过交接、重观察或安全停止。架构变更必须先创建 Issue，说明官方依据、影响范围、
   回退方案，并由 A（项目负责人/Supervisor/集成）确认。
3. `main` 始终保持可运行。禁止直接向 `main` 推送，所有变更均通过 Pull Request（PR）合入。
4. 模型权重、原始/生成数据集、仿真录屏、训练缓存、密钥和个人环境文件默认不得进入 Git。

## 2. 六个角色及默认评审关系

| 角色 | 主责 | 涉及其接口时必须请求其评审 |
|---|---|---|
| A | 项目管理、需求与评分冻结、Supervisor、状态机、令牌、固定生命周期、集成 | Supervisor、跨模块契约、最终集成 |
| B | Isaac Sim、双 Franka、夹爪、控制器、物理与仿真性能 | 仿真环境、执行器、双臂安全状态机 |
| C | 场景、资产、数据与教师轨迹 | 数据格式、场景资产、数据切分 |
| D | VLA 数据与服务迁移支持 | 双臂 VLA 数据和接口评审 |
| E | π0.5/openpi、训练/推理服务与动作适配 | 双臂 π0.5 服务及动作输出 |
| F | 测试、复现、评测、日志、CI、材料与证据链 | 测试、指标、日志、复现和提交材料 |

成员的 GitHub 用户名确认后，再由 A 更新 `.github/CODEOWNERS`。在此之前，PR 作者需在 PR 中手动请求对应角色评审。

## 3. 标准工作流

每个任务必须先有 Issue，再建短生命周期分支，再提交 PR：

```text
Issue → 从最新 main 建分支 → 小步提交 → 自测 → Push → Draft PR
      → CI → 角色评审 → 修改 → Approval → Squash merge → Issue/看板完成
```

### 3.1 分支命名

格式：`<type>/<issue号>-<角色>-<简短说明>`，全部使用小写英文、数字和连字符。

```text
feature/123-a-agent-fsm
fix/124-b-sim-timeout
data/125-c-canonical-split
experiment/126-e-pi05-dual-arm
docs/127-f-evidence-index
hotfix/128-e-action-contract
```

允许的 `type`：`feature`、`fix`、`data`、`experiment`、`docs`、`test`、`refactor`、`hotfix`。
禁止长期个人分支、无 Issue 分支以及直接在他人分支上开发；确需协作时，在 Issue 中指定唯一分支负责人。

### 3.2 提交消息

采用 `type(scope): summary`：

```text
feat(agent): add retry transition for execution timeout
fix(sim): clamp gripper command to safe range
test(contract): cover malformed VLA response
docs(api): document task status error codes
```

常用 `type`：`feat`、`fix`、`test`、`docs`、`refactor`、`perf`、`build`、`ci`、`chore`。
一次提交只处理一个逻辑主题，不提交调试垃圾、缓存、大文件或密钥。

### 3.3 最小命令流程

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/123-a-agent-fsm

# 开发和自测后
git status
git diff
git add <明确的文件路径>
git diff --cached
git commit -m "feat(agent): add task retry state"
git push -u origin feature/123-a-agent-fsm
```

Push 后创建 Draft PR，填写 `.github/PULL_REQUEST_TEMPLATE.md` 的全部适用项。不要用 `git add .` 代替提交前检查。

## 4. PR 合入门槛

PR 必须同时满足：

- 关联 Issue（使用 `Closes #123`）且范围与 Issue 一致；
- 无模型权重、数据集、密钥、个人路径、缓存和无关格式化；
- 新行为有测试，测试/格式检查通过；无法自动测试时提供可复现的手工证据；
- 接口、配置、运行方式或数据格式变化已同步更新文档；
- 至少 1 名非作者批准；跨模块契约、冻结架构或官方指标相关变更需要 A 和受影响模块负责人共同确认；
- 所有评审对话已解决，CI 通过，分支已同步最新 `main`；
- 给出风险、兼容性影响和回退方式。

默认使用 **Squash and merge**。合入后删除远程分支并把 Issue/Project 状态改为 `Done`。

## 5. Definition of Done

任务只有在下列事项全部完成后才算 Done：

1. 验收标准逐项满足，并有日志、截图、测试报告或演示记录可核验；
2. 代码、配置、接口文档和测试一起进入 `main`；
3. CI 通过，必要的跨模块联调完成；
4. 不引入未登记的 P0/P1 缺陷或高风险技术债；
5. 产物可由另一名成员按文档复现；
6. PR 已合入，Issue 已关闭，看板和交付证据链接已更新。

## 6. 安全底线

- 禁止提交 Token、密码、SSH 私钥、云密钥、Webhook、`.env`、内部下载凭证或含敏感参数的日志。
- 发现泄露时，第一步是立即吊销/轮换凭据并通知 A；仅删除文件或提交不等于消除泄露。
- GitHub 普通 Git 对象存在 100 MB 硬限制。模型/数据默认不入库；Git LFS 也只有在 A 评估配额、下载带宽和比赛交付方式并批准后才能使用。
- 禁止在共享历史执行 `git push --force`、`git reset --hard` 或随意改写历史。公共提交出错时使用 `git revert`。

## 7. 需要帮助时

不要独自“硬修”冲突或误提交。停止继续 Push，在对应 Issue/PR 留下：

- 执行的命令；
- 完整报错（先去除敏感信息）；
- `git status` 输出；
- 预期结果与实际结果；
- 是否有未提交的本地工作。

随后请求 A 或熟悉 Git 的成员协助。详细恢复步骤见协作指南。
