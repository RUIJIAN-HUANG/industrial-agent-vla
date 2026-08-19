# B - Isaac Sim 仿真平台盘点

> 历史证据说明：本页记录的是 V1 四工件静态场景在 2026-07-27 的 G0 平台证据，
> 不是当前 V2 八工件场景的验收结论。V2 已通过仓库内静态合同检查，但 GUI、物理、
> IK、抓取和满载搬运仍需在 Isaac Sim 5.1 上重新留证。

> **当前 PR 合并门禁：未通过。** 本文原始 PASS 证据只对应提交
> `b1e6a05fc52ef95a576442bacff96dbe699920b9`，不能证明 PR #7 当前 head。
> 当前脚本已要求 13 个 Prim，并新增 Isaac Sim 5.1 版本、相机像素质量和
> 显式 reset 参数校验。必须在 Linux Isaac Sim 5.1 上对待合并 head 重新
> 执行 `scripts/run_g0_linux.sh`，上传新的三次冷启动、13/13、commit 与
> SHA256 证据，再由 F 复核、A 签署。
>
> 重跑时必须显式设置
> `EXPECTED_GIT_SHA=$(git rev-parse HEAD)`，且工作树必须保持 clean；
> 脚本会在启动 Isaac Sim 前强制核对这两个条件。
>
> 日期：2026-07-27
>
> 执行人：成员 B
>
> 状态：`B PASS - 自动验收与 GUI 人工复核均通过，待 F/A 签署`

## 1. 冻结结论

| 项目 | 结果 |
|---|---|
| 主仿真平台 | NVIDIA Isaac Sim 5.1.x |
| 主机用途 | 双 Franka、三 RGB 相机、单箱交接场景的开发与验证 |
| 主机是否可用 | 可用；G0 自动验收三次独立启动均通过 |
| 回退机 | 暂无回退机 |
| 是否保留 Gazebo 生产路径 | 否 |
| Isaac Sim 根目录 | `/home/xyz/isaacsim` |
| 资产来源 | 在线 NVIDIA Asset Root |

## 2. 主机硬件与系统

以下内容从本次证据目录中的 `platform-inventory.txt` 摘录，不凭记忆填写。

| 项目 | 实际值 | 验证命令 |
|---|---|---|
| 主机名 | `xyz` | `hostname` |
| Linux 发行版 | Ubuntu 22.04.5 LTS (Jammy Jellyfish) | `cat /etc/os-release` |
| 内核 | Linux 6.8.0-111-generic | `uname -a` |
| CPU | 13th Gen Intel Core i9-13900K，24 核/32 线程 | `lscpu` |
| 内存 | 125 GiB（采集时可用 119 GiB） | `free -h` |
| GPU | NVIDIA GeForce RTX 3090 Ti | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| 显存 | 24564 MiB | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` |
| NVIDIA 驱动 | 580.76.05（`nvidia-smi` 显示 CUDA 13.0） | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| 可用磁盘 | 385 GiB（根分区 984 GiB，已用 59%） | `df -h .` |
| CPU 架构 | `x86_64` | `uname -m` |

## 3. Isaac Sim 与项目版本

| 项目 | 实际值/证据 |
|---|---|
| Isaac Sim 版本 | 5.1.0 |
| 安装方式 | Workstation 独立安装目录（非容器） |
| `python.sh` | `/home/xyz/isaacsim/python.sh` |
| Franka USD | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` |
| 仓库 commit | `b1e6a05fc52ef95a576442bacff96dbe699920b9` |
| V1 证据场景合同 | `simulation/configs/single_bin_scene_v1.json` |
| 当前 V2 场景合同 | `simulation/configs/single_bin_scene_v2.json`（本页历史 G0 未覆盖） |
| 物理步长 | `1/120 s`，以配置实际值为准 |
| 渲染步长 | `1/30 s`，以配置实际值为准 |
| 控制频率 | `60 Hz`，以配置实际值为准 |

## 4. 兼容性与风险

请根据真实输出勾选。

- [x] `nvidia-smi` 正常，未出现驱动通信错误。
- [ ] `isaac-sim.compatibility_check.sh` 可运行。
- [x] Isaac Sim GUI 能启动并进入 `app ready`。
- [x] `python.sh` 能运行本仓库独立 Python 脚本。
- [x] Franka USD 能从在线 Asset Root 解析。
- [x] 运行期间不依赖个人密钥或写入仓库的凭据。
- [x] 已记录联网依赖；最终演示前需验证断网加载。

已知问题：

1. Franka 当前来自在线 NVIDIA Asset Root；最终演示前应准备本地资产并完成断网复测。
2. Isaac Sim 5.1 启动日志会扫描部分测试扩展并报告缺少 `psutil`，但不影响本次 G0，三个进程均以 `0` 退出且证据状态均为 `PASS`。

回退方法：

1. 首次编译 shader 较慢时等待完成，不把超时直接当成代码错误。
2. 用户配置损坏时，先备份日志，再使用 Isaac Sim 的 `--reset-user`。
3. 资产在线加载失败时，改为显式传入本地 Franka USD：

   ```bash
   "$ISAAC_SIM_ROOT/python.sh" simulation/run_g0_acceptance.py \
     --evidence-dir artifacts/g0/manual \
     --franka-usd /absolute/path/to/franka.usd
   ```

## 5. 证据索引

| 证据 | 路径/链接 | 结论 |
|---|---|---|
| 自动盘点 | `artifacts/g0/20260727-210649/platform-inventory.txt` | 已采集 |
| 三次启动汇总 | `artifacts/g0/20260727-210649/restart-summary.tsv` | 三次退出码均为 `0` |
| 文件哈希 | `artifacts/g0/20260727-210649/SHA256SUMS.txt` | 已生成 |
| 原始证据压缩包 | `member-b-g0-20260727-210649.tar.gz`，SHA256 `0eb8806c062e58edb44655f2892ef11760de9eb862ef25048abb3487bb1240c1` | 上传到 Draft PR #7 |
| Draft PR | [#7](https://github.com/RUIJIAN-HUANG/industrial-agent-vla/pull/7) | 待 F/A 复核 |
| Issue | 未创建 | 可由 A/F 决定是否需要 |

## 6. B 的签字

- 盘点完成时间：2026-07-27 21:07（Asia/Shanghai）
- 结论：`PASS（成员 B）`
- 阻塞与需要 A/F 协助的事项：请 F 复核本报告与 GUI 截图；请 A/F 确认最终演示的本地资产/断网方案。
