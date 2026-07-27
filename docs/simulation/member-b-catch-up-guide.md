# 成员 B：7 月 24-27 日补交操作手册

这份手册面向第一次使用 Isaac Sim 的成员。目标不是完成抓取算法，而是在老师的
Linux 电脑上补齐 D00-D03 的平台与最小场景证据。

## 0. 完成标准

全部完成时，你应拥有：

1. 已填写的 `docs/simulation/platform-inventory.md`；
2. 已填写的 `docs/simulation/platform-gate.md`；
3. 一份 `artifacts/g0/<时间>/` 原始证据目录；
4. 三次 Isaac Sim 独立启动记录；
5. 1000 步 headless 日志、20 次 Reset 记录；
6. 三张相机样本和一份双臂观测 JSON；
7. GUI 截图或短视频；
8. 一个 Issue、一个个人分支和一个 Draft PR。

## 1. 登录 Linux 电脑并确认仓库

打开 Terminal，执行：

```bash
cd /老师电脑上/industrial-agent-vla
git status
git branch --show-current
git rev-parse HEAD
```

如果仓库还没克隆：

```bash
git clone <老师给你的仓库地址>
cd industrial-agent-vla
```

创建自己的任务分支，不要直接修改 `main`：

```bash
git switch -c feat/b-g0-isaac-platform
```

若分支已经存在：

```bash
git switch feat/b-g0-isaac-platform
```

## 2. 找到 Isaac Sim 5.1

老师若告诉你安装路径，直接使用。常见路径是：

```bash
ls -l ~/isaacsim/python.sh
ls -l /opt/isaacsim/python.sh
```

假设实际路径是 `/opt/isaacsim`：

```bash
export ISAAC_SIM_ROOT=/opt/isaacsim
test -x "$ISAAC_SIM_ROOT/python.sh"
test -x "$ISAAC_SIM_ROOT/isaac-sim.sh"
```

两条 `test` 没有输出且退出码为 0，才继续：

```bash
echo $?
```

不要使用系统的 `python` 运行仿真脚本。Isaac Sim 5.1 应使用它自己的
`python.sh`，这样才能设置正确的扩展路径和动态库。

## 3. 先做兼容性检查

```bash
cd "$ISAAC_SIM_ROOT"
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

保存终端输出：

```bash
mkdir -p /tmp/isaac-b-check
./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window \
  2>&1 | tee /tmp/isaac-b-check/compatibility-check.log
```

若首次启动时间很长，多数情况是在编译 shader。等待进程正常结束，不要重复点击
启动。

## 4. 本地合同预检

回到仓库，用普通 Python 即可：

```bash
cd /老师电脑上/industrial-agent-vla
python3 simulation/scene_layout.py \
  --config simulation/configs/single_bin_scene_v1.json
python3 -m unittest tests.test_scene_layout -v
```

期望看到：

- 所有距离检查为 `[PASS]`；
- 5 个单元测试为 `OK`；
- 提示仍需 Isaac Sim IK/碰撞验证是正常的，不是失败。

## 5. 执行一键 G0 验收

在仓库根目录运行：

```bash
export ISAAC_SIM_ROOT=/你的/isaacsim/实际路径
bash scripts/run_g0_linux.sh
```

脚本会：

1. 自动保存 OS、CPU、内存、GPU、驱动、磁盘和 Git commit；
2. 第一次启动 Isaac Sim，生成冻结 USD；
3. 运行 1000 个 headless 步；
4. 连续执行 20 次 Reset；
5. 读取两台 Franka 的关节名、位置和速度；
6. 从三台相机各保存一张 PPM；
7. 完全关闭 Isaac Sim 后再冷启动两次；
8. 为全部证据生成 SHA-256。

成功时终端最后显示：

```text
G0 AUTOMATED CHECKS PASSED.
```

证据位于：

```text
artifacts/g0/YYYYMMDD-HHMMSS/
```

`artifacts/` 被 Git 忽略，这是正确行为；原始大日志和图像不应直接进入普通 Git
历史。

## 6. 如果一键脚本失败

先打开失败那次的日志，不要重新安装一切：

```bash
less artifacts/g0/<时间>/restart-1/console.log
cat artifacts/g0/<时间>/restart-1/run_result.json
```

常见情况：

### 6.1 找不到 `python.sh`

```text
ERROR: Isaac Sim python.sh is not executable
```

处理：

```bash
export ISAAC_SIM_ROOT=/正确的/isaacsim/路径
bash scripts/run_g0_linux.sh
```

### 6.2 找不到 Franka USD

先找本地资产：

```bash
find "$ISAAC_SIM_ROOT" -path '*Franka*' -name 'franka.usd' 2>/dev/null
```

如果找到了，先单独运行：

```bash
"$ISAAC_SIM_ROOT/python.sh" simulation/run_g0_acceptance.py \
  --evidence-dir artifacts/g0/manual-franka \
  --franka-usd /找到的/绝对路径/franka.usd
```

若本地没有，需让老师确认是否已安装 Local Assets Pack，或者该电脑是否允许访问
NVIDIA 在线 Asset Root。

### 6.3 相机图全黑

先确认 GPU 与驱动：

```bash
nvidia-smi
```

然后用 GUI 打开生成场景，等待 shader 编译完成再观察。不要为了“通过”而删除相机
验收。

### 6.4 出现 NaN、物体弹飞或离开工作区

保留失败证据，不把报告写成 PASS。将以下文件附到 Issue：

```text
restart-1/console.log
restart-1/run_result.json
restart-1/reset_report.json
```

这属于 P0，通知 A 和 F。

## 7. GUI 人工复核

自动脚本通过后运行 GUI：

```bash
cd "$ISAAC_SIM_ROOT"
./isaac-sim.sh
```

在 Isaac Sim 中：

1. 选择 `File -> Open`；
2. 打开仓库中的
   `simulation/generated/single_bin_scene_v1.usda`；
3. 在 Stage 树检查：

   ```text
   /World/Robots/Arm_A
   /World/Robots/Arm_B
   /World/Parts/P01 ... P04
   /World/Bins/Bin_01
   /World/Stations/PACK_STATION
   /World/Stations/HANDOFF_CENTER
   /World/Stations/FINISHED_01
   /World/Cameras/CAM_A_TOP
   /World/Cameras/CAM_HANDOFF
   /World/Cameras/CAM_B_TOP
   ```

4. 点击 Play，运行至少 30 秒；
5. 检查机器人不是空壳，物体没有穿模、弹飞；
6. 分别切换三台 Camera，确认视野覆盖正确区域；
7. 截图或录制 20-30 秒短视频；
8. 停止仿真并重新打开 USD，确认外部引用仍完整。

自动脚本不能代替这一步。

## 8. 打开三张相机样本

Linux 文件管理器通常可以直接打开 PPM。也可运行：

```bash
xdg-open artifacts/g0/<时间>/restart-1/cameras/CAM_A_TOP.ppm
xdg-open artifacts/g0/<时间>/restart-1/cameras/CAM_HANDOFF.ppm
xdg-open artifacts/g0/<时间>/restart-1/cameras/CAM_B_TOP.ppm
```

如果老师电脑没有桌面环境，把这三个文件复制到自己的电脑查看。

## 9. 填写两份报告

根据真实证据填写：

```text
docs/simulation/platform-inventory.md
docs/simulation/platform-gate.md
```

不可填写的内容写 `BLOCKED`，不要猜。特别注意：

- 只有三次退出码都是 0 才能写连续三次启动通过；
- 只有 `headless_steps_completed=1000` 才能写 1000 步通过；
- 只有实际打开三张图才能勾相机人工检查；
- GUI 没检查时，Gate 仍保持 `DRAFT`；
- 日志失败时不得删除失败记录。

## 10. 提交 GitHub

只提交代码、脚本和填写后的 Markdown，不提交整个 `artifacts/`：

```bash
git status
git add docs/simulation simulation scripts
git diff --cached
git commit -m "test(sim): add Isaac Sim G0 evidence and report"
git push -u origin feat/b-g0-isaac-platform
```

然后在 GitHub：

1. 创建 Draft PR；
2. 关联成员 B 的 G0 Issue；
3. PR 描述中写运行命令；
4. 上传小型必要截图，或填写校内证据盘链接与 SHA；
5. 请 F 复核证据，请 A 做 Gate 决策。

## 11. 这次补交不包含什么

以下不是 7 月 24-27 日补交的完成条件：

- π0.5 或 OpenVLA 训练；
- YOLO 微调与 mAP；
- 完整抓取、纠姿、装箱控制器；
- 最终一镜到底比赛视频；
- 真机验证。

它们是后续里程碑。当前只需要先证明平台、场景、两臂和三相机可稳定运行。
