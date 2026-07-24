# GitHub 与六人团队协作指南

> 适用项目：`RUIJIAN-HUANG/industrial-agent-vla`
> 适用对象：A–F 六名团队成员，按“第一次使用 GitHub 也能完成一次规范 PR”编写
> 推荐主路径：**GitHub Desktop + GitHub 网页**；文中同时给出完整 Git CLI 等价命令
> 核心原则：先 Issue、后分支；小步提交；PR 评审；`main` 永远可运行；权重和数据默认不进 Git

## 1. 先理解四个概念

| 概念 | 在本项目中的含义 | 新手最容易犯的错误 |
|---|---|---|
| Repository（仓库） | 项目文件及完整变更历史 | 把仓库当网盘，直接上传大文件 |
| Branch（分支） | 为一个 Issue 隔离开发的工作区 | 六个人都直接改 `main` |
| Commit（提交） | 一次可说明、可回退的逻辑变更 | 几天工作挤成一个“update”提交 |
| Pull Request（PR） | 把分支变更交给团队检查并合入 | 未自测、无 Issue、无证据就请求合入 |

团队的标准闭环：

```text
Issue（目标/验收）
  ↓
从最新 main 创建任务分支
  ↓
开发 → 本地检查 → 小步 Commit → Push
  ↓
Draft PR → CI → Review → 修改
  ↓
Ready for review → Approval → Squash merge
  ↓
删除分支 → 关闭 Issue → Project 移到 Done
```

## 2. 工具选择：推荐 GitHub Desktop

### 2.1 为什么以 Desktop 为主

GitHub Desktop 能清楚显示“当前分支、哪些文件被改动、每一行差异、是否已 Push”，最适合六名 Git 新手统一操作。GitHub 网页用于 Issue、Project、PR 和 Review；CLI 用于可复现命令、服务器开发以及故障排查。

团队约定：

- 日常改代码：优先 GitHub Desktop；
- 任务、看板、PR、Review：GitHub 网页；
- 远程服务器、自动化、复杂冲突：使用本指南中的 CLI；
- 不使用 GitHub 网页的 `Add file > Upload files` 上传代码、目录、数据或权重；
- 网页直接编辑只限很小的文档修正，并且仍要创建新分支和 PR。

### 2.2 三种方式对应关系

| 目的 | GitHub Desktop | CLI | GitHub 网页 |
|---|---|---|---|
| 克隆仓库 | `File > Clone repository` | `git clone <URL>` | 复制 `Code > HTTPS` 地址 |
| 更新本地 | `Fetch origin` / `Pull origin` | `git pull --ff-only origin main` | 不替代本地更新 |
| 新建分支 | `Current branch > New branch` | `git switch -c <branch>` | 可为 Issue 创建分支 |
| 查看改动 | `Changes` | `git status`、`git diff` | PR 的 `Files changed` |
| 提交 | 左下角 Summary + `Commit` | `git commit` | 仅适合小文档 |
| 推送 | `Push origin` | `git push -u origin <branch>` | 不适用 |
| 创建 PR | `Create Pull Request` | `gh pr create` | `Compare & pull request` |
| 评审 | 打开网页 | `gh pr review` | `Files changed > Review changes` |

## 3. 第一次加入项目

### 3.1 账号与权限

1. 注册/登录 GitHub，建议开启双因素认证。
2. 把自己的 GitHub 用户名发给 A，由 A 邀请进入仓库；不要把密码或 Token 发给任何人。
3. 接受仓库邀请后，确认可以看到 Issues、Pull requests 和 Projects。
4. 在 GitHub 账户中验证邮箱；若不想公开邮箱，可使用 GitHub 提供的 `noreply` 邮箱。
5. 安装 GitHub Desktop。CLI 用户还需安装 Git；可选安装 GitHub CLI（命令为 `gh`）。

### 3.2 Git 首次配置

在 PowerShell、Git Bash 或终端执行一次：

```bash
git config --global user.name "你的显示名称"
git config --global user.email "你在 GitHub 已验证的邮箱或 noreply 邮箱"
git config --global init.defaultBranch main
git config --global pull.ff only
git config --global core.autocrlf true

git config --global --list
```

Windows 推荐 `core.autocrlf true`，Linux/macOS 推荐：

```bash
git config --global core.autocrlf input
```

不要把比赛组织 Token 写入 `git config`、脚本或仓库。HTTPS 登录优先使用 Git Credential Manager 弹出的浏览器授权。

### 3.3 使用 HTTPS 克隆

#### GitHub Desktop

1. 仓库网页点击 `Code > Local > HTTPS`，确认地址为
   `https://github.com/RUIJIAN-HUANG/industrial-agent-vla.git`。
2. GitHub Desktop 选择 `File > Clone repository > URL`。
3. 粘贴地址，选择一个**不在微信/网盘同步目录内**的本地路径。
4. 点击 `Clone`，完成后确认顶部 `Current branch` 是 `main`。
5. 点击 `Fetch origin`；如显示 `Pull origin`，继续点击 Pull。

#### CLI

```bash
git clone https://github.com/RUIJIAN-HUANG/industrial-agent-vla.git
cd industrial-agent-vla
git remote -v
git branch --show-current
git status
```

期望看到：

- `origin` 的 fetch/push 地址都指向官方团队仓库；
- 当前分支为 `main`；
- 工作区为 `working tree clean`。

如果 HTTPS 询问“密码”，不要填写 GitHub 账户密码；应使用浏览器授权或有最小权限和有效期的 Personal Access Token。

## 4. 每日开工前：先同步，再开发

### 4.1 每日 5 分钟检查

每位成员开工时完成：

1. 看 GitHub Project 的 `Ready / In Progress / Blocked`；
2. 确认自己只有 1 个主要任务处于 `In Progress`；
3. 阅读相关 Issue 的最新验收标准和接口变更；
4. 同步 `main`；
5. 再切回自己的任务分支并合并 `main`。

### 4.2 GitHub Desktop 操作

如果本地还有未提交改动，先提交一个完整的 WIP commit，或使用
`Repository > Stash All Changes`。不要在状态不清楚时切换分支。

1. `Current branch` 选择 `main`。
2. 点击 `Fetch origin`，随后点击 `Pull origin`。
3. 切回自己的任务分支。
4. 选择 `Branch > Update from main`。
5. 若无冲突，继续开发；若有冲突，按第 10 节处理。

### 4.3 CLI 等价命令

```bash
# 先确认当前状态
git status

# 工作区干净后更新 main
git switch main
git pull --ff-only origin main

# 回到任务分支，并把最新 main 合进来
git switch feature/123-a-agent-fsm
git merge main
git push
```

`--ff-only` 会在历史异常时停止，而不会偷偷生成一次不明合并。任务分支一旦已 Push，团队默认使用 `merge main` 同步，不要求新手 rebase 和强制推送。

## 5. 从 Issue 开始任务

### 5.1 Issue 必须写清什么

使用仓库的 `Task / 任务` 表单，至少填：

- 责任角色 A–F；
- 目标和业务价值；
- 来自官方 PDF/冻结设计的依据；
- 明确的 In Scope / Out of Scope；
- 可核验的交付物；
- 依赖项与阻塞；
- Definition of Done；
- 预计工作量和目标日期。

“研究一下”“把模型做完”“优化代码”都不是可执行任务。更好的例子：

> D：给 OpenVLA 推理服务增加 `/health` 与 `/v1/actions:predict`，按 Agent 接口文档返回 request_id、状态和动作数组；提供 3 个契约测试，D6 18:00 前提交 Draft PR。

### 5.2 Issue 拆分标准

一个 Issue 应当：

- 由一名主要负责人在 0.5–2 个工作日内完成；
- 只产生一个主要 PR；
- 验收结果可以判断“通过/不通过”；
- 若预计超过 2 天，拆成接口、实现、测试/证据等子 Issue；
- 被阻塞时，立即设置 `Blocked`，写明阻塞人、所需输入和最迟解除时间。

## 6. 分支规范

格式：

```text
<type>/<issue号>-<角色>-<简短英文说明>
```

示例：

```text
feature/123-a-agent-fsm
fix/124-b-controller-timeout
data/125-c-canonical-split
experiment/126-d-openvla-oft
experiment/127-e-pi05-action-head
test/128-f-contract-regression
docs/129-f-evidence-index
hotfix/130-a-router-fallback
```

规则：

- 全部小写，只用字母、数字、`/` 和 `-`；
- 分支必须从**最新 `main`**创建；
- 一个分支只服务一个 Issue；
- 默认只有一个负责人 Push；结对开发需在 Issue 中明确；
- PR 合并后删除分支；
- 禁止 `dev-A`、`test2`、`mybranch`、`final-final` 等不可追踪名称。

### 6.1 Desktop 创建分支

1. 更新 `main`。
2. `Current branch > New branch`。
3. 输入规范名称。
4. `Create branch based on: main`。
5. 点击 `Publish branch`。

### 6.2 CLI 创建分支

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/123-a-agent-fsm
git push -u origin feature/123-a-agent-fsm
```

不要为了“备份”同时创建 `feature-x-v2`、`feature-x-final`。Git 本身已经保存历史。

## 7. 开发、检查、Commit、Push

### 7.1 提交前四次检查

```bash
# 1. 我在哪个分支？
git branch --show-current

# 2. 哪些文件发生变化？
git status

# 3. 具体改了什么？
git diff

# 4. 精确加入暂存区后，再检查一次
git add path/to/file1 path/to/file2
git diff --cached
```

确认无误后：

```bash
git commit -m "feat(agent): add explicit retry transition"
git push
```

不推荐直接 `git add .`。如果确实使用，必须在 Commit 前逐文件查看 `git diff --cached`。

### 7.2 Commit 粒度

好的 Commit：

- 能用一句话说明；
- 单独检出后仍能通过相应检查；
- 代码与对应测试一起提交；
- 不混入自动格式化其他模块、个人配置和临时数据。

格式：

```text
type(scope): imperative summary
```

示例：

```text
feat(agent): add fallback route for VLA timeout
fix(openvla): validate camera frame dimensions
test(contract): reject unknown task status
docs(sim): document headless launch command
ci(python): run contract tests on pull requests
```

### 7.3 Desktop 提交

1. 左侧 `Changes` 逐文件勾选；不该提交的取消勾选。
2. 点击文件查看红/绿差异。
3. 在 `Summary` 写规范 Commit 标题。
4. 必要时在 `Description` 写原因、限制和 Issue。
5. 点击 `Commit to <branch>`。
6. 点击 `Push origin`。

### 7.4 不完整工作如何保存

优先把工作收敛为可说明的小 Commit。确实无法提交时：

```bash
git stash push -u -m "wip issue-123 before syncing main"
git switch main
git pull --ff-only origin main
git switch feature/123-a-agent-fsm
git stash pop
```

`stash` 只是短期本地缓冲，不是备份。离开电脑前，应将不含密钥/大文件的可恢复工作 Push 到任务分支。

## 8. 创建 Pull Request

### 8.1 什么时候先开 Draft PR

完成首个可评审骨架后就开 Draft PR，不必等到最后一天。Draft PR 可以让接口消费者提前发现偏差。

特别是以下变更，应尽早开 Draft：

- Agent ↔ VLA 请求/响应契约；
- Agent ↔ 仿真执行与安全状态接口；
- 数据 schema、坐标系、单位、时间戳；
- 官方指标解释、评测入口和证据格式；
- 冻结架构可能受影响的实现。

### 8.2 网页/Desktop 创建 PR

1. Push 后在 Desktop 点击 `Create Pull Request`，会打开 GitHub 网页。
2. 确认 `base: main`，`compare: 你的任务分支`。
3. 标题使用：`[角色][类型] 简短结果`，例如
   `[A][Feature] 实现 Agent 超时重试状态转换`。
4. 完整填写 PR 模板；正文写 `Closes #123`。
5. 未完成时选择 `Create draft pull request`。
6. 请求对应角色评审。
7. 自测、文档、CI 都完成后点击 `Ready for review`。

### 8.3 GitHub CLI 创建 PR

首次使用：

```bash
gh auth login
gh auth status
```

创建 Draft PR：

```bash
gh pr create \
  --base main \
  --head feature/123-a-agent-fsm \
  --title "[A][Feature] 实现 Agent 超时重试状态转换" \
  --draft \
  --web
```

查看状态：

```bash
gh pr status
gh pr checks --watch
gh pr view --web
```

模板应说明：

- 为什么改，而不只是改了什么；
- 官方/冻结需求依据；
- 接口、配置和兼容性影响；
- 自测命令及结果；
- 截图、日志、报告或演示证据；
- 风险与回退方法；
- 尚未完成的项。

## 9. Review、CI 与合入

### 9.1 谁来 Review

| 变更范围 | 最少评审人 |
|---|---|
| 单模块普通变更 | 1 名非作者，优先该模块备份角色或 F |
| Agent 与跨模块契约 | A + 至少 1 名受影响模块负责人 |
| 仿真/执行安全 | B + A 或 F |
| 数据 schema/切分 | C + 消费数据的 D/E/F 之一 |
| OpenVLA/π0.5 动作契约 | D/E 对应负责人 + A/B 中的消费者 |
| 官方指标、评测、证据 | F + A |
| 冻结架构或官方要求解释 | A + 受影响负责人；必须回链官方依据 |

没有配置 CODEOWNERS 前，作者必须手动邀请这些评审人。

### 9.2 Review 检查清单

评审人按顺序检查：

1. PR 是否解决关联 Issue，是否越界；
2. 是否违反官方指标或冻结架构；
3. 接口 schema、单位、坐标系、错误码、超时和幂等是否明确；
4. 失败路径、安全状态和回退是否覆盖；
5. 测试是否真正验证验收标准；
6. 是否混入密钥、权重、数据、缓存或个人绝对路径；
7. 文档和复现命令是否足够另一人执行。

评论使用可执行表达：

```text
Blocking：该分支把角度按 degree 返回，但接口约定为 rad。请统一单位并补契约测试。
Suggestion：可将重复校验提取为函数；不阻塞本 PR，可另建技术债 Issue。
Question：重试期间 request_id 是否保持不变？请在接口文档明确。
```

作者不应只回复“已改”，应回复变更位置、验证方式并解决对话。

### 9.3 CI 规则

`.github/workflows/ci.yml` 会：

- 检出代码并使用 Python 3.11；
- 校验官方 PDF、冻结图和初版方案的 5 个 SHA-256；
- 存在 Python 文件时运行 `ruff format --check` 和 `ruff check`；
- 存在 `tests/` 下的测试时，根据 `requirements-ci.txt`、`requirements-dev.txt`、`pyproject.toml` 或 `requirements.txt` 安装项目依赖并运行 `pytest`；
- 自动校验 JSON Schema、默认配置和核心契约测试；
- 若某个未来分支暂时没有 Python/测试，则明确显示 Skip，不因空目录误报失败；
- 自动读取当前 `pyproject.toml` 中的测试依赖与工具配置。

若项目后续依赖 Isaac Sim、CUDA 或大模型，不应把它们直接塞进基础 CI。基础 CI 只跑 CPU 可承受的单元/契约测试；GPU/仿真测试另建按需工作流并记录环境。

### 9.4 Definition of Done

合入前逐项满足：

- Issue 验收标准全部有证据；
- 代码、配置、接口文档和测试同步更新；
- CI 通过，必要联调通过；
- 无未登记 P0/P1 问题；
- 另一成员可根据文档复现；
- 至少所需评审人批准；
- 所有对话解决；
- 给出回退方式；
- PR 使用 Squash and merge；
- 合入后删除分支、关闭 Issue、Project 置为 Done。

## 10. 冲突处理

### 10.1 先判断是否应自己解决

只解决自己理解的文件。若冲突涉及：

- 官方 PDF、冻结架构；
- 其他角色负责的接口；
- 大片自动生成文件；
- 数据 schema、坐标系或安全策略；

先在 PR 里 `@` 相关负责人，不要猜测保留哪一边。

### 10.2 Desktop 处理

1. 在任务分支执行 `Branch > Update from main`。
2. Desktop 显示冲突文件后，点击 `Open in Visual Studio Code` 或编辑器。
3. 查找：

   ```text
   <<<<<<<
   =======
   >>>>>>>
   ```

4. 理解两边意图，手工形成最终内容；不能简单“全部接受当前/传入”。
5. 保存并确认所有标记已删除。
6. 回 Desktop 标记 resolved，创建 merge commit。
7. 运行测试后 Push。

随时可以在 Commit 前点击 `Abort merge`，回到冲突前状态。

### 10.3 CLI 处理

```bash
git switch feature/123-a-agent-fsm
git status
git merge main

# 出现冲突后
git status
# 手工编辑冲突文件
git add path/to/resolved-file
git diff --check
git commit
git push
```

如果不确定：

```bash
git merge --abort
git status
```

然后把 `git status` 和冲突文件列表发到 PR 请求帮助。不要在慌乱时执行 `reset --hard` 或 `push --force`。

## 11. 安全撤销与恢复

### 11.1 只撤销未暂存的某个文件

先预览：

```bash
git diff -- path/to/file
```

确认后：

```bash
git restore path/to/file
```

这会丢弃该文件未提交的修改，无法从普通 Git 历史找回，务必先确认。

### 11.2 从暂存区移除，但保留本地修改

```bash
git restore --staged path/to/file
```

### 11.3 修改“尚未 Push”的最后一次 Commit

仅当该 Commit 还没 Push、且分支无人共享时：

```bash
git add path/to/missed-file
git commit --amend
```

若已经 Push，不要 amend 后强推；再建一个修正 Commit。

### 11.4 撤销已经 Push/合入的公共 Commit

```bash
git switch main
git pull --ff-only origin main
git switch -c fix/234-a-revert-bad-change
git revert <commit-sha>
git push -u origin fix/234-a-revert-bad-change
```

然后创建 PR。`git revert` 会增加一个反向提交，不会改写共享历史。

### 11.5 找回误删分支或 Commit

停止继续操作：

```bash
git reflog
git branch recovery/issue-123 <找到的-commit-sha>
```

请 A 或熟悉 Git 的成员确认后再 Push。

### 11.6 禁止新手自行执行

除非 A 明确确认目标和备份，不执行：

```text
git reset --hard
git clean -fd
git push --force
git push --force-with-lease
git filter-repo
git rebase 已共享的分支
```

这些命令可能永久丢失本地工作或重写团队历史。

## 12. 大文件、模型、数据与 Git LFS

### 12.1 本项目默认策略

以下内容默认不进入 Git，也不应通过 GitHub 网页上传：

- `.pt`、`.pth`、`.ckpt`、`.safetensors`、`.onnx`、`.bin` 等模型权重；
- 原始/增强/生成数据集、RLDS 数据、相机录制和教师轨迹；
- Isaac/仿真缓存、Docker 镜像、conda/venv；
- 视频、长日志、训练 checkpoint、TensorBoard/W&B 本地缓存；
- 包含授权限制或个人信息的数据。

Git 中只保存：

- 下载/生成脚本；
- 小型、脱敏、许可明确的测试 fixture；
- 配置、schema、数据清单；
- 外部存储位置说明、版本、SHA-256、大小、许可证和生成方法；
- 可复现实验元数据和摘要报告。

GitHub 对普通 Git 对象有 **100 MB 硬限制**。不要把 99 MB 当安全目标；仓库克隆会携带历史，多个中等二进制同样会拖垮协作。

### 12.2 LFS 不是默认答案

Git LFS 会把大文件内容放在 LFS 存储，但仍受存储量、带宽和下载配额影响。比赛中的六人频繁拉取权重/数据，很容易消耗配额。因此：

1. 先使用团队批准的外部对象存储/网盘/模型仓库；
2. 用清单记录版本和 SHA-256；
3. 只有 A 确认配额、成本、网络、许可证和最终交付方式后，才能启用 LFS；
4. 未经批准，不执行 `git lfs track`，更不允许先提交再讨论。

获批后的参考流程：

```bash
git lfs install
git lfs track "*.approved-extension"
git add .gitattributes
git add path/to/approved-file
git commit -m "build(assets): track approved artifact with lfs"
git lfs ls-files
git push
```

LFS 只改变存储方式，不解决数据许可、隐私、版本治理或配额问题。

### 12.3 大文件误提交但尚未 Push

停止 Push，并请求 A 协助。可先从暂存区移除并保留本地文件：

```bash
git restore --staged path/to/large-file
```

如果已经 Commit 但尚未 Push，不要随意尝试历史清理，把 `git status` 和 Commit SHA 发给 A。若已 Push，则需要团队级历史清理和所有成员重新同步，必须统一安排。

## 13. 密钥与敏感信息防护

禁止提交：

- GitHub Token、Hugging Face Token、API Key、密码；
- SSH 私钥、云服务凭据、Webhook secret；
- `.env`、个人配置、浏览器 Cookie；
- 含 Token 的命令历史、截图、日志或 Notebook 输出；
- 内部 IP、临时签名下载链接和数据授权凭证。

提交前检查：

```bash
git status
git diff --cached
git grep -n -I -E "BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|api[_-]?key|access[_-]?token|secret"
```

该检查不能替代专业扫描，但能发现明显问题。未来可在仓库设置中启用 Secret scanning 和 Push protection。

一旦泄露：

1. **立即吊销/轮换凭据**，不要先花时间删 Commit；
2. 通知 A，并说明泄露位置和时间；
3. 删除公开日志/Artifact；
4. 由 A 评估历史清理、GitHub 支持和全员重新克隆；
5. 建立事故 Issue（不得把真实密钥贴进去）。

仅仅删除文件、Force Push 或关闭仓库，均不能保证密钥失效。

## 14. Issues 与 Projects 看板

### 14.1 推荐看板列

| 状态 | 进入条件 | 离开条件 |
|---|---|---|
| Backlog | 有价值但未排期 | A 完成优先级和范围确认 |
| Ready | 依赖已满足、验收清晰、可立即领取 | 负责人开工 |
| In Progress | 已有负责人和任务分支 | 开 Draft PR 或确认阻塞 |
| In Review | Draft/正式 PR 已存在 | 合入、退回开发或阻塞 |
| Blocked | 无法靠负责人独立推进 | 写明解除条件后转回进行中 |
| Done | PR 合入且 DoD 全满足 | 不再移动；回归问题新建 Bug |

### 14.2 推荐字段

- `Role`：A/B/C/D/E/F；
- `Priority`：P0/P1/P2/P3；
- `Day / Milestone`：D1–D40 或具体里程碑；
- `Estimate`：0.5d/1d/2d；
- `Component`：agent/simulation/data/openvla/pi05/test/docs/integration；
- `Risk`：Low/Medium/High；
- `Target date`：目标完成日期；
- `Evidence`：PR、日志、视频、报告链接。

### 14.3 WIP 与优先级

- 每人同时最多 1 个主要 `In Progress`，另可有 1 个轻量 Review；
- P0：当天阻断 Demo/安全/官方硬指标，立即处理；
- P1：阻断里程碑或两人以上，24 小时内处理；
- P2：普通计划任务；
- P3：优化或技术债，不挤占核心交付；
- Blocked 超过 4 个工作小时必须在 Issue 通知 A；
- 每日结束前更新状态、完成比例、证据和次日第一步。

### 14.4 Issue 与 PR 的关联

PR 正文写：

```text
Closes #123
```

只有 PR 合入默认分支后，Issue 才会自动关闭。若一个 PR 仅完成部分内容，写 `Refs #123`，不要提前关闭 Issue。

## 15. 仓库设置建议（由 A 配置）

### 15.1 `main` 分支保护

在 GitHub 仓库 `Settings > Branches` 或 Rulesets 中，对 `main` 设置：

- Require a pull request before merging；
- Require at least 1 approval；
- Dismiss stale approvals when new commits are pushed；
- Require review from Code Owners（填好真实用户名后再启用）；
- Require status checks：`Python quality and tests`；
- Require conversation resolution before merging；
- Require branch to be up to date before merging；
- Block force pushes；
- Block branch deletion；
- 限制绕过规则；管理员也尽量走 PR；
- 仅开放 Squash merge，关闭不需要的合入方式。

如果 GitHub 套餐或仓库权限不支持某项，A 在项目风险登记中记录替代控制，例如“双人网页确认 + CI 截图”。

### 15.2 标签建议

```text
type:task        type:bug         type:docs
priority:P0      priority:P1      priority:P2      priority:P3
role:A ... role:F
status:blocked   needs-review
component:agent  component:sim    component:data
component:openvla component:pi05  component:test
```

标签用于筛选，不替代 Issue 中的验收标准。

### 15.3 CODEOWNERS

`.github/CODEOWNERS` 现在只有注释占位，因为不能猜测成员 GitHub 用户名。A 收齐用户名后：

1. 用真实 `@username` 替换示例；
2. 确认用户名对仓库有读权限；
3. 提 PR 验证自动请求 Reviewer；
4. 再启用 “Require review from Code Owners”。

## 16. 紧急 Hotfix

“很赶”不等于可以直接 Push `main`。Hotfix 只用于 P0：Demo 无法运行、安全风险、官方硬指标回归、主分支完全阻塞。

### 16.1 标准 Hotfix

1. 创建 P0 Bug Issue，写明影响、复现、负责人和回退点；
2. 从最新 `main` 创建 `hotfix/<issue>-<role>-<summary>`；
3. 只做最小修复，不顺手重构；
4. 增加至少一个回归测试或可重复的手工验证；
5. PR 标题加 `[HOTFIX]`，写清风险和回退 Commit；
6. 作者不是 A 时：A + 相关模块负责人确认；A 是作者时：相关负责人 + F 确认；
7. CI 通过后 Squash merge；
8. 合入后立即执行烟雾测试并更新证据。

命令示例：

```bash
git switch main
git pull --ff-only origin main
git switch -c hotfix/501-a-router-deadlock
# 最小修改、测试、提交
git push -u origin hotfix/501-a-router-deadlock
gh pr create --base main --title "[HOTFIX][A] 修复路由死锁" --web
```

若必须回退已合入变更，优先通过 `git revert` 新建 PR。只有当 GitHub 本身故障且比赛交付即将截止时，A 才能记录紧急绕过；至少需要另一名成员实时确认，并在恢复后 2 小时内补齐 Issue、PR、测试与事故复盘。

## 17. 常见问题速查

### `git pull` 提示 divergent branches

不要选择随机策略。确认工作区干净，然后：

```bash
git status
git pull --ff-only origin main
```

仍失败则停止，把输出发给 A；通常说明本地 `main` 被错误提交，需要保留/迁移后恢复。

### Push 被拒绝 `non-fast-forward`

可能远程分支有新提交：

```bash
git fetch origin
git status
git log --oneline --decorate --graph --all -20
```

如果这是你的独占分支，合并远程变更：

```bash
git merge origin/你的分支名
git push
```

若不是独占分支，先联系分支负责人。不要 Force Push。

### Commit 到了 `main`，尚未 Push

停止操作，记录 `git status` 和 `git log -3 --oneline`，请 A 协助把 Commit 搬到新分支。不要先 Reset。

### Commit 作者邮箱错误

先改未来配置：

```bash
git config --global user.email "正确邮箱"
```

已 Push 的历史不为“好看”而重写。若贡献归属有实质影响，由 A 统一处理。

### PR 中出现大量无关换行变化

停止提交，确认 `core.autocrlf` 和编辑器 EOL 设置；只保留目标文件的必要差异。不要以“格式化”为由重写整个他人模块。

### CI 在本地通过、GitHub 失败

查看失败步骤和 Python 版本，复制 CI 中的命令本地执行。Issue/PR 中贴失败日志的关键部分（去除敏感信息），不要只贴“CI 红了”。

## 18. 每日协作节奏（建议）

| 时间点 | 每名成员 | A（项目负责人） |
|---|---|---|
| 开工 | 更新看板、同步 main、确认当日 DoD | 发布/确认今日 Ready 任务与依赖 |
| 中途 | 首个可评审骨架开 Draft PR | 优先处理接口和阻塞问题 |
| 阻塞 4h | Issue 转 Blocked，写解除条件 | 指定解阻人和最迟反馈时间 |
| 收工前 | Push、更新 Issue、附证据、写次日第一步 | 看 WIP、PR 队列、P0/P1 和里程碑偏差 |

每日状态更新模板：

```text
今日完成：
- #123：完成 Agent retry 状态与 3 个测试，PR #45

证据：
- 测试命令/结果：
- 日志或截图链接：

阻塞/风险：
- 需要 B 在明日 10:00 前确认 timeout 单位

明日第一步：
- 根据评审修改并联调 OpenVLA mock
```

## 19. 官方 GitHub 参考

- [Cloning a repository（HTTPS Clone）](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
- [Pull requests](https://docs.github.com/en/pull-requests)
- [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [About Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [Push protection](https://docs.github.com/en/code-security/concepts/secret-security/push-protection)
