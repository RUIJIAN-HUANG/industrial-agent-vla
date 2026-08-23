# 成员 B：Isaac Sim 5.1 键采前冒烟验收

## 结论与边界

本入口用于验证成员 B 的以下交付物：

- 键盘指令严格映射到冻结的 7 维动作顺序
  `[dx, dy, dz, dax, day, daz, gripper]`；
- 三个固定相机 `CAM_A_TOP`、`CAM_HANDOFF`、`CAM_B_TOP` 均输出
  1280×720 RGB，并写入共享 CAS；
- 每个动作前后均产生新 observation ID、单调时间戳和完整 camera 字段；
- 动作经过 `IsaacExecutionEnvironment`、主线程 Gate、Franka Controller 和
  safe-stop 回读。

脚本输出的是 `smoke_only` JSONL 证据，不是 Canonical Episode，不能直接拿去
训练。正式训练数据仍需成员 C 的 Canonical Recorder、回放与 split manifest。

## 第一次运行：只做 Arm_A 的一个小动作

先关闭其他正在运行的 Isaac Sim 实例，再在 Linux 终端执行：

```bash
cd "$HOME/Sceneconstruction/industrial-agent-vla"
git fetch origin
git switch main
git pull --ff-only

export ISAAC_SIM_ROOT="$HOME/isaacsim"
"$ISAAC_SIM_ROOT/python.sh" -m pytest -q \
  tests/test_keyboard_teleop.py \
  tests/test_isaac_rgb_pipeline.py \
  tests/test_image_cas.py \
  tests/test_isaac_runtime.py \
  tests/test_isaac_environment.py \
  tests/test_isaac_franka_controller.py

"$ISAAC_SIM_ROOT/python.sh" simulation/run_keyboard_teleop_smoke.py \
  --arm-id Arm_A \
  --max-actions 1 \
  --artifact-dir artifacts/keyboard-teleop-smoke/arm-a-first
```

窗口和场景稳定后，在启动脚本的终端中输入 `q` 并按 Enter。它只请求 Arm_A
沿 robot-base `+Z` 移动 5 mm。观察机械臂没有异常后输入 `p` 并按 Enter保存
检查点，再输入 `x` 并按 Enter安全退出。

仅当终端最后出现以下字段时，本轮通过：

```json
{
  "status": "PASS",
  "action_count": 1,
  "three_rgb_cas_streams": true,
  "online_observation_validated": true,
  "safe_stop_confirmed": true
}
```

未完成任何动作就直接按 `x` 或关闭窗口时，本轮必须返回 `FAIL`，不能作为 RGB、
Observation 或控制链路的有效验收证据。

还要人工确认：只有 Arm_A 动作、Arm_B 保持静止；没有穿模、飞出、明显抖动；
画面不是全黑；Stage 中不存在 `/Environment/defaultLight`，只使用冻结场景的
`/World/Lighting`。

## 第二次运行：扩展到键盘小步操作

第一轮通过后，把 `--max-actions` 改为 10。键位如下：

| 输入 | 动作 |
|---|---|
| `w` / `s` | robot-base `+X` / `-X`，每次 5 mm |
| `a` / `d` | robot-base `+Y` / `-Y`，每次 5 mm |
| `q` / `e` | robot-base `+Z` / `-Z`，每次 5 mm |
| `i` / `k` | rotation-vector X 正/负 5° |
| `j` / `l` | rotation-vector Y 正/负 5° |
| `u` / `o` | rotation-vector Z 正/负 5° |
| `g` | 切换夹爪开/合；写入的是目标端点 1/0，不是增量 |
| `r` | 重置场景 |
| `p` 或输入 `space` | 保存一个冒烟检查点 |
| `x` | safe-stop 并退出 |

每次只输入一个键，等终端打印 `after=<observation_id>` 后再输入下一键。任何
异常都立即输入 `x`；如果界面失去响应，则终止进程并保留 artifact 与 command
journal，不得删除 journal 后继续同一轮。

## 第三次运行：Arm_B

Arm_A 的 1 步和 10 步测试都通过后，新建独立 artifact 目录运行 Arm_B：

```bash
"$ISAAC_SIM_ROOT/python.sh" simulation/run_keyboard_teleop_smoke.py \
  --arm-id Arm_B \
  --max-actions 10 \
  --artifact-dir artifacts/keyboard-teleop-smoke/arm-b-first
```

## 通过后仍缺的正式采集前提

成员 B 的两臂键采冒烟通过，只能证明仿真控制和在线观测边界可用。开始正式、
大规模采集之前还必须补齐：

| 成员 | 必需交付物 |
|---|---|
| C | Canonical Recorder；Episode schema 实装；成功/失败/恢复轨迹；replay；split registry 与 manifest |
| D | Canonical 到 RLDS/OpenVLA-OFT 的转换器，用一条真实 Episode 完成读取验收 |
| E | Canonical 到 LeRobot 的转换器和 norm stats，用同一条真实 Episode 完成读取验收 |
| F | 图像、状态、动作、时间戳、NaN/Inf、哈希、GT 隔离和回放一致性的自动 QA |
| A（组长） | 冻结采集版本、批准试采轮数与成功/失败保留规则，并验收上述交付物 |

正确顺序是：成员 B 两臂冒烟 → C 接入 Recorder → 共同采 1 条 Canonical
Episode → C 回放 → D/E 分别试读 → F QA → 组长批准小批试采 → 小批验收通过后
才扩大规模。

## Isaac 窗口内直接键控与录屏

终端模式需要输入字母后按 Enter。录屏或实际人工示教应改用 GUI 模式：

```bash
"$ISAAC_SIM_ROOT/python.sh" simulation/run_keyboard_teleop_smoke.py \
  --input-mode gui \
  --arm-id Arm_A \
  --max-actions 10 \
  --artifact-dir artifacts/keyboard-teleop-smoke/gui-arm-a
```

出现 `Keyboard Teleop` 浮窗后，点击一次 Isaac viewport，再轻按按键，不需要
Enter。只处理 `KEY_PRESS`，长按产生的 repeat 事件不会变成额外动作。浮窗显示
当前机械臂、键位、动作数和执行状态。`X` 或 `Esc` 请求 safe-stop 并退出。

GUI 模式沿用 W/A/S/D 等键。操作时不要按住鼠标右键，否则 Isaac viewport 的
fly-navigation 也可能响应这些键；每次只轻按一个键，等待浮窗显示 `COMPLETE`
后再继续。
