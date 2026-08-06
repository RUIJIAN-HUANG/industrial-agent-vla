# 成员 B 长期进度与协作日志

> 用途：记录成员 B 已完成的工作、证据、当前阻塞和下一步，供成员 B 与 Codex 在整个项目周期内持续使用。  
> 项目周期：2026-07-25 至 2026-09-02（之后为提交缓冲期）。  
> 最后更新：2026-07-30。  
> 重要原则：只有代码、测试、日志、截图、压缩包、PR 评论或审核记录支持的事项，才能标记为“完成”。

## 1. 每次协作的固定流程

每次开始新任务时，成员 B 和 Codex 必须按以下顺序工作：

1. 先完整阅读本文件。
2. 查看仓库当前分支、HEAD、工作区状态和远端更新。
3. 查看最新的每日计划、组长消息、PR 审核意见和 CI 状态。
4. 区分以下三种状态：
   - **技术执行完成**：代码或实验已经在本机完成。
   - **证据已提交**：证据已经正式出现在 GitHub PR/Issue 中，而不是只停留在预览页面。
   - **团队验收完成**：所需审核人已经复核、签字或批准。
5. 一次只推进一个主任务；遇到失败先保存日志并定位原因，不盲目重复运行。
6. 每次任务结束时更新本文件的“完成记录”“当前状态”和“下一步”。

如果开启新的 Codex 对话，成员 B 应先告诉 Codex：

```text
请先读取 docs/project-management/member-b-progress-log.md，再继续我的新任务。
```

## 2. 成员 B 的固定背景

| 项目 | 当前信息 |
|---|---|
| 角色 | 成员 B |
| 当前经验 | 科研与 Isaac Sim 新手，需要分步骤指引 |
| 固定职责 | Isaac Sim、双 Franka/夹爪、控制器、物理、headless、共享区防碰撞 |
| 日常对话设备 | Windows |
| Isaac Sim/最终仿真设备 | 老师提供的 Linux 工作站 |
| Windows 仓库 | `D:\机械臂\industrial-agent-vla` |
| Linux 仓库 | `/home/xyz/Sceneconstruction/industrial-agent-vla` |
| Isaac Sim 根目录 | `/home/xyz/isaacsim` |
| Isaac Sim 版本 | 5.1.0 |
| Linux 系统 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GitHub 仓库 | `RUIJIAN-HUANG/industrial-agent-vla` |
| 成员 B 的 PR | PR #7 |
| PR 分支 | `feat/b-g0-isaac-platform` |
| 协作边界 | Codex 可以分析和修改本地文件，但不替成员 B 推送、合并或批准 PR |

## 3. 当前总状态

### 3.1 当前代码基线

| 项目 | 值 |
|---|---|
| PR #7 最后取证提交 | `8ddeec19bc8f70ae1187ca58f7dc8d7789fe595a` |
| PR #7 合入 `main` 的提交 | `09fcfbc` |
| 合并提交说明 | `[G0][Member B] Isaac Sim 5.1 platform automated acceptance (#7)` |
| 当前远端 `main` | `211152c`（包含 PR #7 和随后合入的 PR #9） |
| Windows 旧本地分支 | `fix/pr7-isaac-version-api`，远端跟踪分支已删除 |
| Linux 旧本地分支 | `feat/b-g0-isaac-platform`，不得继续推送 |

PR #7 使用 squash 方式合并，因此 `8ddeec1` 不会显示为 `main` 的直接祖先；
合并后的等价代码位于 `09fcfbc`，其中已包含模块级 `get_version()` 修复。

### 3.2 PR #7 当前状态

- 最新提交对应的 Linux Isaac Sim 5.1 G0 强校验：**技术执行完成**。
- 最终证据压缩包：**已生成并通过校验**。
- PR #7：**已合并到 `main`**。
- 合并提交：`09fcfbc`。
- 旧远端分支：**已删除**。
- G0 代码集成状态：**完成**。

远端仓库事实已经取代组长较早截图中的 `CHANGES_REQUESTED / a065728`
状态。当前准确表述是：

```text
成员 B 的 PR #7 已以 09fcfbc 合入 main，G0 平台代码集成完成。
```

## 4. 已完成工作记录

### 4.1 2026-07-25 至 2026-07-27：G0 最小场景与验收链路

完成内容：

- 建立双 Franka、桌面、料箱、圆柱零件、区域标记和三路相机的最小 Isaac Sim 场景。
- 建立 Linux G0 自动运行脚本。
- 建立 1000 steps、20 resets、3 次启动和三路相机取证流程。
- 建立平台清单、Gate 文档、追赶指南和运行结果 JSON。
- 在 Isaac Sim 5.1 中实际打开生成的 USD 场景并检查 Stage 树。
- 确认场景外观无明显穿模；播放后双臂产生动作并停止。
- 生成三张相机图、机器人观测、reset 报告、控制台日志和证据哈希。

相关主要提交：

| 提交 | 作用 |
|---|---|
| `aa3f61f` | 增加成员 B G0 验收脚本和文档 |
| `4e1aad8` | 兼容 Isaac Sim 5.1 的 Stage 创建返回值 |
| `a6928a9` | reset 后刷新 Stage 句柄 |
| `b1e6a05` | 兼容 Isaac Sim 5.1 articulation DOF 名称 |
| `8e3fbda` | 格式化 G0 脚本 |
| `02fc077` | 记录 GUI 检查和 PR 证据 |

最初证据对应旧提交 `b1e6a05`，后来因 PR 头提交变化而被组长要求重新取证。旧证据只保留作排障历史，不能作为当前最终证据。

### 4.2 2026-07-28：加强 G0 Stage 合同和主分支兼容

完成内容：

- 将最新 `main` 合入 PR 分支并处理兼容问题。
- 加强 `create_new_stage()` 返回值检查，避免误用残留旧 Stage。
- 增加 Stage 对象类型验证。
- 增加 Z-up、米制、公斤制的写入与回读验证。
- 增加 G0 冻结站点验证：
  - `PACK_STATION`
  - `HANDOFF_CENTER`
  - `FINISHED_01`
- 增加相应的单元测试。

相关提交：

| 提交 | 作用 |
|---|---|
| `db101f1` | 合并当时最新的 `origin/main` |
| `6ed02ae` | 加强 Isaac G0 Stage 验证 |
| `bc5a0b1` | 加强 Isaac Sim 5.1 G0 证据 Gate |
| `a065728` | 再次合并远端主线形成审核基线 |

`6ed02ae` 主要修改：

- `simulation/isaac_compat.py`
- `simulation/run_g0_acceptance.py`
- `simulation/single_bin_scene_builder.py`
- `tests/test_g0_acceptance.py`
- `tests/test_isaac_compat.py`

### 4.3 2026-07-29：修复 Isaac Sim 5.1 版本 API 并重跑最终 G0

第一次在提交 `a065728` 上运行新的 G0 Gate 时失败：

```text
AttributeError: 'Version' object has no attribute 'get_version'
```

定位结果：

- Isaac Sim 5.1 的 `get_version()` 是 `isaacsim.core.version` 的模块级函数。
- 不能使用 `Version().get_version()`。

完成修复：

- `simulation/isaac_compat.py`
  - 改为 `from isaacsim.core.version import get_version`
  - 直接调用 `get_version()`
- `tests/test_isaac_compat.py`
  - 测试改为模拟模块级 `get_version`
  - 保留 5.1 接受、其他版本拒绝和扩展启用失败等验证

相关提交：

```text
8ddeec1 fix(sim): use Isaac module version API
```

测试记录：

- 兼容性单元测试：12 tests，全部通过。
- 完整 Python 测试：275 passed，1 个非阻断弃用警告。
- Ruff format：通过。
- Ruff check：通过。
- 官方基线验证：通过。
- 项目冻结输入验证：通过。
- 仓库卫生检查：通过。
- `git diff --check`：通过。

### 4.4 2026-07-30：完成 Isaac Lab 键盘遥操作可行性尝试

任务来源：

- 组长询问成员 B 是否试过“键采”。
- 本次仅验证键盘遥操作可行性，不是正式批量数据采集。

环境确认：

| 项目 | 结果 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2（扩展启动日志） |
| 独立 Conda 环境 | `mylab_env` |
| Python | 3.11.15 |
| Isaac Lab 源码目录 | `/home/xyz/IsaacLab` |
| Isaac Sim 链接 | `/home/xyz/IsaacLab/_isaac_sim -> /home/xyz/isaacsim` |

安装与排障：

- 排除旧的 `leisaac` 环境：该环境实际为 Isaac Lab 2.1.0、Isaac Sim 4.5。
- 确认 `mylab_env` 能载入 Isaac Sim 5.1 Python 模块。
- 修复 `flatdict==4.0.1` 构建时缺少 `pkg_resources` 的问题。
- 将 `source/isaaclab` 以 editable 方式安装到 `mylab_env`。
- 核心包验证输出为 `Isaac Lab core OK`。
- 使用 `AppLauncher` 成功启动 Isaac Sim 5.1 空场景，终端输出
  `[INFO]: Setup complete...`。

键盘遥操作命令：

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard
```

实际验证结果：

- [x] 机械臂能够通过键盘控制移动。
- [x] `K` 能切换夹爪开合。
- [x] `R` 能重置环境。

当前结论：

```text
成员 B 已完成 Isaac Sim 5.1 + Isaac Lab 2.3.2 下的键采可行性 smoke。
键盘输入、末端控制、夹爪控制和环境重置链路均正常。
```

数据记录追加验证：

- 已运行 Isaac Lab 官方 `record_demos.py`，完成 1 条成功示范。
- 终端明确输出 `Recorded 1 successful demonstrations.`。
- HDF5 文件位于
  `/home/xyz/IsaacLab/datasets/member_b_keyboard_smoke_20260730.hdf5`。
- 该文件属于官方 Franka 堆叠任务的键采 smoke 数据，不是比赛自建场景的
  正式交付数据，也尚未上传 GitHub。

边界说明：

- 本次没有修改 `industrial-agent-vla` 比赛仓库代码。
- 本次没有创建提交、推送或 PR。
- `mylab_env` 仍有若干强化学习训练包的版本冲突；它们未阻断本次键盘
  遥操作 smoke，但正式采集前应使用冻结环境或单独清理。

## 5. 最新 G0 最终证据

### 5.1 运行基线

| 项目 | 结果 |
|---|---|
| Git SHA | `8ddeec19bc8f70ae1187ca58f7dc8d7789fe595a` |
| `EXPECTED_GIT_SHA` | 与实际 SHA 完全一致 |
| Isaac Sim | 5.1.0 |
| 第一次运行 | PASS |
| Headless steps | 1000/1000 |
| Resets | 20/20 |
| Reset settle steps | 120 |
| 第二次独立冷启动 | PASS |
| 第三次独立冷启动 | PASS |
| 三次退出码 | 全部为 0 |
| 最终输出 | `G0 AUTOMATED CHECKS PASSED` |

### 5.2 相机证据

| 相机 | 分辨率 | 自动检查 |
|---|---:|---|
| `CAM_A_TOP` | 1280×720 | 非纯色检查通过 |
| `CAM_HANDOFF` | 1280×720 | 非纯色检查通过 |
| `CAM_B_TOP` | 1280×720 | 非纯色检查通过 |

三张图片均来自本次 `restart-1`，并记录独立像素 SHA256；`online_gt_included = false`。

### 5.3 Linux 证据位置

```text
/home/xyz/Sceneconstruction/industrial-agent-vla/artifacts/g0/20260729-051650
```

最终压缩包：

```text
/home/xyz/g0-deliverables/member-b-g0-20260729-051650-8ddeec1.tar.gz
```

外部校验文件：

```text
/home/xyz/g0-deliverables/member-b-g0-20260729-051650-8ddeec1.tar.gz.sha256
```

GitHub 兼容的校验文件副本：

```text
/home/xyz/g0-deliverables/member-b-g0-20260729-051650-8ddeec1.tar.gz.sha256.txt
```

完整性结果：

- 证据目录内 `SHA256SUMS.txt` 所有文件校验成功。
- `SHA256 verification exit code: 0`。
- 最终 `.tar.gz` 压缩包外部 SHA256 校验成功。

## 6. 已发现并解决的关键兼容问题

| 问题 | 原因 | 处理 |
|---|---|---|
| 找不到 `scene_layout.py` | 在 Isaac Sim 安装目录而非仓库目录执行 | 切换到仓库根目录运行 |
| 没有 `run_result.json` | 旧脚本异常退出但外层脚本误报成功 | 强化 Gate，缺少结果文件直接失败 |
| `SetStageUpAxis` 参数错误 | Stage 创建 API 返回布尔值时误当 Stage 使用 | 严格获取并校验当前 Stage |
| CustomData 数组类型错误 | USD 字典不接受普通 Python list | 使用 `Vt.DoubleArray`/`Vt.StringArray` |
| `GetPrimAtPath` 参数错误 | Isaac 5.1 需要 `Sdf.Path` | 增加路径兼容转换 |
| 没有 `joint_names` | Isaac 5.1 articulation 使用 DOF 名称接口 | 增加 DOF 名称兼容读取 |
| `Version().get_version()` 不存在 | `get_version` 是模块函数 | 在 `8ddeec1` 中改用模块级 API |

这些错误日志属于正常排障历史，不得把失败目录伪装成最终通过证据。

## 7. 当前待办

### P0：完成 PR #7 审核闭环

- [x] 最新提交对应的 Linux Isaac Sim 5.1 实机证据已生成。
- [x] 完整证据压缩包和相机图已生成。
- [x] PR #7 已由团队合入 `main`，合并提交为 `09fcfbc`。
- [x] 旧远端功能分支已删除。

不得再向旧 PR #7 分支推送。后续键采 smoke 或 D05 工作必须从最新
`origin/main` 新建独立分支。

### 后续主任务候选

按照当前 40 天计划，2026-07-29（D05）成员 B 的后续方向是：

```text
脚本实现单圆柱“到达—夹取—抬升—放置”，
并让总 Agent FSM 接入 dummy 环境；
目标交付包括脚本教师 v0、20 回合日志和 Agent mock trace。
```

但开始前必须先查看当天最新任务公告和组长消息；如果计划或接口已经更新，以最新仓库证据和组长要求为准。

### 2026-07-29：组长询问“是否试过键采”

组长发来的采集方式建议为：

- 状态机 + IK 自动采集：80%–90%，用于稳定、成功的标准抓取轨迹；
- 键盘人工采集：10%–20%，用于补充纠偏、重新抓取、失败恢复数据；
- SpaceMouse/手柄：有条件再使用，用于更连续的六自由度控制。

当前判断：

- 成员 B 确实承担“脚本专家/控制器运行与示范轨迹生产”的工作，因此参与数据采集；
- 成员 B 不负责数据标注策略、Canonical Recorder、Split 或 Manifest 的最终所有权；
- 键采是补充采集方式，不应替代状态机 + IK 的主采集链路；
- 组长问“有没有试过键采”目前应视为键采可行性确认，尚不能仅凭该问句认定正式批采数量和 DoD；
- 当前仓库未发现现成的 keyboard/teleop、脚本教师或 Canonical Episode Recorder 实现，不能直接开始正式批采。
- 2026-07-30 已完成 Isaac Lab 官方 Franka 堆叠环境的键盘遥操作
  可行性 smoke：机械臂移动、夹爪切换和环境重置均正常。
- 该 smoke 只证明控制链路可用，尚未产生可提交的正式 Episode 数据。

开始键采前需要组长或当日 Issue 明确：

1. 控制 `Arm_A`、`Arm_B` 还是先只控制一只机械臂；
2. 末端增量控制还是逐关节控制；
3. 按键映射、控制频率、速度/角速度限幅和夹爪开合键；
4. 是否已经有 C 负责的 Canonical Recorder，以及输出目录和 Schema；
5. 本次只是 1 条 smoke，还是要求若干有效/失败/恢复 Episode；
6. 数据对应的场景 Git SHA、固定 Seed 和验收标准。

## 8. 常用安全命令

### Windows 查看状态

```powershell
& "C:\Program Files\Git\cmd\git.exe" status -sb
& "C:\Program Files\Git\cmd\git.exe" log -3 --oneline
```

### Linux 查看状态

```bash
cd "$HOME/Sceneconstruction/industrial-agent-vla"
git status -sb
git branch --show-current
git rev-parse HEAD
```

### Linux 设置 Isaac Sim

```bash
export ISAAC_SIM_ROOT="$HOME/isaacsim"
export EXPECTED_GIT_SHA="$(git rev-parse HEAD)"
```

### 禁止在没有明确授权时使用

```text
git push --force
git reset --hard
git clean -fd
```

## 9. 后续更新模板

每次完成新任务后，在“已完成工作记录”中追加：

```markdown
### YYYY-MM-DD：任务名称

- 任务来源：
- 开始时 Git SHA：
- 修改文件：
- 执行命令：
- 测试结果：
- 证据位置：
- PR/Issue：
- 当前状态：技术完成 / 证据已提交 / 团队已验收
- 尚未完成：
- 新发现的问题与解决方法：
```
