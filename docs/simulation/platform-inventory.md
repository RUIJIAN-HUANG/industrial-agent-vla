# B - Isaac Sim 仿真平台盘点

> 日期：待填写
>
> 执行人：成员 B
>
> 状态：`DRAFT - 必须用老师 Linux 电脑的真实输出替换所有“待填写”`

## 1. 冻结结论

| 项目 | 结果 |
|---|---|
| 主仿真平台 | NVIDIA Isaac Sim 5.1.x |
| 主机用途 | 双 Franka、三 RGB 相机、单箱交接场景的开发与验证 |
| 主机是否可用 | 待填写 |
| 回退机 | 待填写；若没有，明确写“暂无回退机” |
| 是否保留 Gazebo 生产路径 | 否 |
| Isaac Sim 根目录 | 待填写，例如 `/home/<user>/isaacsim` |
| 资产来源 | 待填写：本地 Asset Pack / 在线 NVIDIA Asset Root |

## 2. 主机硬件与系统

以下内容从本次证据目录中的 `platform-inventory.txt` 摘录，不凭记忆填写。

| 项目 | 实际值 | 验证命令 |
|---|---|---|
| 主机名 | 待填写 | `hostname` |
| Linux 发行版 | 待填写 | `cat /etc/os-release` |
| 内核 | 待填写 | `uname -a` |
| CPU | 待填写 | `lscpu` |
| 内存 | 待填写 | `free -h` |
| GPU | 待填写 | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| 显存 | 待填写 | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` |
| NVIDIA 驱动 | 待填写 | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| 可用磁盘 | 待填写 | `df -h .` |
| CPU 架构 | 待填写 | `uname -m` |

## 3. Isaac Sim 与项目版本

| 项目 | 实际值/证据 |
|---|---|
| Isaac Sim 版本 | 5.1.x，补充完整版本号 |
| 安装方式 | 待填写：Workstation ZIP / 容器 / Python 包 |
| `python.sh` | 待填写绝对路径 |
| Franka USD | 从 `restart-1/run_result.json` 复制 `franka_asset` |
| 仓库 commit | 从 `platform-inventory.txt` 的 `[git]` 部分复制 |
| 场景合同 | `simulation/configs/single_bin_scene_v1.json` |
| 物理步长 | `1/120 s`，以配置实际值为准 |
| 渲染步长 | `1/30 s`，以配置实际值为准 |
| 控制频率 | `60 Hz`，以配置实际值为准 |

## 4. 兼容性与风险

请根据真实输出勾选。

- [ ] `nvidia-smi` 正常，未出现驱动通信错误。
- [ ] `isaac-sim.compatibility_check.sh` 可运行。
- [ ] Isaac Sim GUI 能启动并打开空 Stage。
- [ ] `python.sh` 能运行本仓库独立 Python 脚本。
- [ ] Franka USD 能从本地资产包或 Asset Root 解析。
- [ ] 运行期间不依赖个人密钥或写入仓库的凭据。
- [ ] 已记录联网依赖；最终演示前需验证断网加载。

已知问题：

1. 待填写；没有则写“本次未发现阻断问题”。

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
| 自动盘点 | `artifacts/g0/<时间>/platform-inventory.txt` | 待填写 |
| 三次启动汇总 | `artifacts/g0/<时间>/restart-summary.tsv` | 待填写 |
| 文件哈希 | `artifacts/g0/<时间>/SHA256SUMS.txt` | 待填写 |
| Draft PR | 待填写 | 待填写 |
| Issue | 待填写 | 待填写 |

## 6. B 的签字

- 盘点完成时间：待填写
- 结论：`PASS / FAIL / BLOCKED`
- 阻塞与需要 A/F 协助的事项：待填写
