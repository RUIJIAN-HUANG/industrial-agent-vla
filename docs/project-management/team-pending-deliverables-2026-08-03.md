# 正式键采与规模化数据采集前：各成员待交付物清单

> 整理日期：2026-08-03  
> 适用仓库：`RUIJIAN-HUANG/industrial-agent-vla`  
> 整理目的：明确正式键采、Canonical 数据生成、模型转换和规模化采集前仍缺少的交付物。  
> 注意：本文件依据 2026-08-03 同步到的远端分支和当前冻结文档整理；“已有”表示仓库中已能看到相关实现，“尚待交付”表示尚未在 `main` 中形成可复现、可验收的完整产物。

## 1. 先说结论

目前已经能够做以下事情：

- 在 Isaac Sim 5.1.0 中启动冻结测试场景；
- 用键盘通过统一 7 维动作接口控制 Arm_A 和 Arm_B；
- 读取三路 1280×720 RGB 相机；
- 将 RGB 写入 CAS，并保存动作轨迹、检查点和测试结果；
- 在 Isaac Sim GUI 视口保持焦点，直接按键控制机械臂；
- 完成了多轮 `smoke_only=true` 的链路验证。

但是，目前还不能把这些烟囱测试产物称为“正式训练数据”，也不应立即开始大规模采集。最关键的第一阻塞项是：

> 成员 C 尚需交付 Canonical Recorder、Replay、Split Registry 和一条可供 D/E/F 共同验收的 Golden Canonical Episode。

当前正确顺序是：

```text
A 冻结合同与验收口径
    → B 完成场景、控制、GUI 键采和 Linux 验证
    → C 接入 Canonical Recorder，生成 Golden Episode
    → D/E/F 分别用同一 Golden Episode 验证转换、Loader 和 QA
    → A 放行小批量采集
    → 小批量回放/转换/QA 全通过
    → 才开始规模化采集
```

## 2. 总览表

| 成员 | 当前主责 | 仓库中已经看到的内容 | 正式采集前最关键的待交付物 | 是否阻塞正式采集 |
|---|---|---|---|---|
| A / 组长 | 合同冻结、集成、放行 | 主线已有冻结架构与接口文档 | Golden Episode 验收口径、版本冻结、分支评审与放行决定 | 是 |
| B / 仿真控制 | Isaac Sim、双臂、相机、键采 | 终端烟囱与 GUI 直接键盘分支；三路 RGB/CAS；Arm_A/Arm_B 烟囱结果 | 完成 GUI 四组 Linux 验证、整理证据；后续接 C Recorder；50 Reset/可达性/碰撞报告 | 是，但当前可继续做 GUI 小规模验证 |
| C / 场景数据 | Canonical 数据、Recorder、Replay、Split | 远端未看到独立 Recorder 实现分支 | Canonical Recorder、Replay、Split、Manifest、Golden Episode | **首要阻塞** |
| D / 皓瀚 | OpenVLA-OFT / Arm_B | `origin/second` 有 Arm_B 数据输入适配器与合成样例测试 | 完整 Canonical→RLDS、真实 Golden Episode Loader、离线 Docker 和来源追踪 | 是 |
| E / 黄浩 | π0.5 / Arm_A | `origin/fix/15-e-pi05-lerobot-data-gate` 有 Canonical Reader、LeRobot/OpenPI 转换与测试 | 真实 Golden Episode 转换、离线遍历、真实 norm stats、来源追踪 | 是 |
| F / YOLO 与 QA | 数据 QA、YOLO/COCO、mAP、证据 | `origin/feature/yolo-runtime` 有独立 YOLO 运行服务 | Canonical QA CLI、YOLO/COCO Exporter、离线 mAP 和数据泄漏检查 | 是 |

## 3. 成员 C：首要待交付物

### 3.1 当前判断

当前远端分支中没有看到可以验收的 Canonical Recorder 实现。文档已经描述了目标格式，但“有格式说明”不等于“已有录制程序”。

### 3.2 C 需要交付的代码

1. **Canonical Recorder**
   - 接收 B 输出的新鲜 Observation、7 维 Action、FSM/阶段事件和终局状态；
   - 每个 Step 必须绑定同一时刻的图像、机器人状态和动作；
   - 录制中断时不能留下看似完整的 Episode；
   - 完成后以原子方式发布整个 Episode。

2. **Canonical Replay**
   - 能从 `meta.json + steps.jsonl + rgb/` 重放或逐步检查；
   - 能发现缺帧、错帧、索引断裂、时间戳倒退、动作维度错误和文件损坏；
   - 回放结果应生成机器可读报告。

3. **Split 与版本工具**
   - 采集前为 Episode 分配 `train/val/test`；
   - 同一 `scenario_group_id/scene_seed` 以及其 Failure/Recovery 必须处于同一 Split；
   - 发布 `split_registry_v1.json`；
   - 禁止采集结束后按图片帧随机切分。

4. **Manifest 与校验工具**
   - 为发布文件生成 SHA-256；
   - 检查文件是否缺失或被修改；
   - 发布 Dataset Card、版本号和来源说明。

### 3.3 单条 Canonical Episode 至少应包含

```text
<episode_id>/
├── meta.json
├── steps.jsonl
├── events.jsonl
├── rgb/
│   ├── CAM_A_TOP/
│   ├── CAM_HANDOFF/
│   └── CAM_B_TOP/
├── terminal_state.json
└── checksums.sha256
```

当前场景没有腕相机，因此模型兼容字段保留，但必须明确写成：

```json
"wrist_image": null
```

### 3.4 `meta.json` 关键字段

至少包括：

- `schema_version`
- `episode_id`
- `scenario_group_id`
- `split`
- `scene_seed`
- `asset_variant`
- `task_id`
- `instruction`
- `robot_role`
- `scene_config_sha256`
- `controller_version`
- `recorder_version`
- `camera_ids`
- `control_hz`
- `render_hz`
- `started_at`
- `ended_at`
- `outcome`
- `dataset_failure_label`
- `parent_episode_id`
- `eligible_for_imitation`

### 3.5 `steps.jsonl` 每步关键字段

至少包括：

- 连续的 `step_index`；
- 单调递增时间戳；
- 唯一且新鲜的 `observation_id`；
- 三路 RGB 相对路径和图像摘要；
- 机器人关节状态；
- `tcp_pose=[x,y,z,qx,qy,qz,qw]`，四元数顺序固定为 `xyzw`；
- 夹爪状态；
- `robot.arm_a_retreated` 与 `robot.arm_b_retreated`；
- 统一 7 维动作 `[dx,dy,dz,dax,day,daz,gripper]`；
- 动作持续时间；
- Agent/FSM 阶段；
- 控制令牌；
- 安全标志；
- `valid_for_training`。

### 3.6 C 的第一份验收产物

C 不需要一开始就批量录几百条。第一步只需交付：

- Recorder 的分支或 PR；
- 1 条 Arm_A Golden Canonical Episode；
- 1 条 Arm_B Golden Canonical Episode，或者一条清晰分段且角色不混淆的完整协作 Episode；
- 对应的 Replay PASS 报告；
- `checksums.sha256`；
- Split 和 Manifest；
- 交给 D/E/F 的读取说明。

### 3.7 C 的验收标准

- [ ] Episode 必需文件完整；
- [ ] Step 索引连续；
- [ ] 时间戳单调且 RGB/状态同步误差在允许范围；
- [ ] 三路 RGB 可解码且尺寸为 1280×720；
- [ ] 状态和动作无 NaN/Inf；
- [ ] Action 最后一维固定为 7；
- [ ] Replay 能从头到尾完成；
- [ ] `checksums.sha256` 全部通过；
- [ ] 在线目录不包含 GT；
- [ ] Golden Episode 可被 D/E/F 在干净环境读取。

## 4. 成员 D：OpenVLA-OFT / Arm_B 待交付物

### 4.1 已有内容

远端 `origin/second` 已经包含：

- `services/openvla_oft/src/openvla_oft/dataset.py`
- `services/openvla_oft/tests/test_dataset.py`

这表明 Arm_B 数据输入适配已经开始，并有合成字典/样例级测试。

### 4.2 尚待交付

1. **完整 Canonical→RLDS 转换器**
   - 读取完整 Episode 目录，而不是只接收手工构造的 Python 字典；
   - 校验 `meta.json`、`steps.jsonl`、图像和 SHA；
   - 正确生成 Episode 边界；
   - 正确设置 `is_first/is_last/is_terminal`；
   - 保留 Canonical `episode_id/step_index`，保证可以反查来源。

2. **Arm_B 角色约束**
   - 只接受 `robot_role=arm_b_openvla`；
   - 只使用 `CAM_B_TOP`；
   - 不得缺少 B 相机后改用其他相机；
   - `wrist_image=null`；
   - 只在 durable `handoff.ready` 后开始；
   - 不得控制 Arm_A。

3. **动作与状态约束**
   - 状态和动作均严格为 7 维；
   - 旋转分量统一采用已冻结的旋转向量增量；
   - 夹爪端点必须与 Canonical 物理含义一致；
   - 拒绝错误角色、错误相机、错误维度、NaN/Inf、坏哈希和错乱时间戳。

4. **真实数据 Loader Smoke**
   - 使用 C 提供的 Arm_B Golden Episode；
   - 从第一步遍历到最后一步；
   - 随机抽查至少 10 个 Step，证明 RLDS 能反查 Canonical 来源；
   - 输出 Loader 日志与机器可读结果。

5. **离线与复现**
   - Docker 中离线加载，不在启动时临时下载依赖或权重；
   - 记录代码 Git SHA、数据 Manifest SHA、模型/预处理配置；
   - 提供一条可复制的运行命令。

### 4.3 D 的验收标准

- [ ] 一条真实 Arm_B Golden Episode 转换成功；
- [ ] RLDS Episode/Step 数量与 Canonical 一致；
- [ ] 指令、图像、状态、动作时间对齐；
- [ ] `is_first/is_last/is_terminal` 正确；
- [ ] Loader 可离线从头到尾遍历；
- [ ] 错误角色/相机/维度/哈希输入会明确失败；
- [ ] 输出中保留数据与代码来源 SHA；
- [ ] Docker 干净环境可以复现。

## 5. 成员 E：π0.5 / Arm_A 待交付物

### 5.1 已有内容

远端 `origin/fix/15-e-pi05-lerobot-data-gate` 已经包含：

- Canonical v1 Reader；
- Canonical→LeRobot/OpenPI 相关转换；
- Loader Smoke；
- norm stats 相关代码；
- 单元测试和格式修正。

这部分基础比“只有文档”更完整，但目前仍主要是合成 Fixture/单元测试，尚缺真实 Golden Episode 验证。

### 5.2 尚待交付

1. **真实 Canonical→LeRobot/OpenPI 转换**
   - 使用 C 提供的 Arm_A Golden Episode；
   - 读取 `meta.json + steps.jsonl + rgb/CAM_A_TOP + checksums.sha256`；
   - 保证 Episode/Step/图像/语言/状态/动作数量一致；
   - 禁止使用 `CAM_HANDOFF` 或 `CAM_B_TOP` 代替 `CAM_A_TOP`；
   - `wrist_image=null`。

2. **Arm_A 角色约束**
   - 只接受 `robot_role=arm_a_pi05`；
   - 原始冻结指令直接进入模型；
   - 只控制 Arm_A；
   - 不得接收 Arm_B 目标、YOLO 结果或在线 GT。

3. **真实 Loader Smoke**
   - 写入 staging；
   - 正常 close/finalize；
   - 重新离线打开；
   - 从头到尾遍历；
   - 随机抽查转换前后动作和来源 ID。

4. **独立 norm stats**
   - 只使用 Train Split；
   - 使用 π0.5/OpenPI 实际预处理后的数据计算；
   - 不得与 OpenVLA 共享统计量；
   - 保存统计结果、配置与数据 Manifest SHA。

5. **离线与复现**
   - Docker 中离线加载；
   - 记录代码 SHA、数据 SHA、转换配置和 norm stats SHA；
   - 提供可复制命令与日志。

### 5.3 E 的验收标准

- [ ] 一条真实 Arm_A Golden Episode 转换成功；
- [ ] LeRobot/OpenPI Episode 和 Step 数量正确；
- [ ] 图像固定使用 `CAM_A_TOP`；
- [ ] 状态/动作严格为 7 维且无 NaN/Inf；
- [ ] 转换后数据可关闭、重开并完整遍历；
- [ ] norm stats 仅由 Train Split 计算；
- [ ] 转换结果可追溯到 Canonical Episode/Step；
- [ ] Docker 干净环境可以离线复现。

## 6. 成员 F：YOLO、Canonical QA 与离线评测待交付物

### 6.1 已有内容

远端 `origin/feature/yolo-runtime` 已经包含独立 YOLO 服务、配置、Dockerfile、接口和服务测试。

这说明“在线调用 YOLO”已有基础，但它不等于“数据 QA、标注导出和 mAP 评测链已经完成”。

### 6.2 尚待交付

1. **Canonical QA CLI**
   - 检查文件和必填字段；
   - 检查 Step 索引连续；
   - 检查时间戳和 Observation ID；
   - 解码图片并核对 1280×720；
   - 校验 SHA-256；
   - 检查状态/动作维度、NaN/Inf；
   - 检查成功、失败、Recovery 和 `valid_for_training`；
   - 检查 Split 泄漏；
   - 发现错误时返回非 0，并输出机器可读报告。

2. **冻结五类类别表**

```text
part_upright
part_inverted
part_fallen
bin_box
bin_slot
```

类别 ID 一旦冻结，后续版本不得静默修改。

3. **YOLO/COCO Exporter**
   - 从同一份隔离的 `offline_gt/` 同时导出 YOLO TXT 和 COCO JSON；
   - 保证 `image_id/annotation_id` 唯一；
   - 检查 bbox 不越界、面积大于 0；
   - 预测 JSON 与 GT JSON 严格分离。

4. **离线 mAP 评测**
   - 保存原始 prediction JSON；
   - 计算 AP50、AP75、mAP50:95；
   - 分类别、分相机报告；
   - 保存漏框、错框、空检测和典型失败图；
   - 记录模型 SHA、数据 SHA、命令和依赖版本。

5. **在线 GT 隔离验证**
   - YOLO 在线容器不能挂载 `offline_gt/`；
   - Supervisor、VLA、在线 Verifier 不能读取仿真 GT、实例 ID 或真实物体坐标；
   - YOLO 超时、空检测、低置信度或坏响应只记录失败，不得阻塞 VLA，也不得控制令牌。

### 6.3 F 的验收标准

- [ ] C 的 Golden Episode 通过 Canonical QA；
- [ ] 人为删除帧、修改哈希、制造 NaN 时 QA 会失败；
- [ ] Split Seed 交集为 0；
- [ ] YOLO TXT 与 COCO JSON 来自同一份 GT；
- [ ] prediction 与 GT 严格分离；
- [ ] AP50/AP75/mAP50:95 可从原始文件重新计算；
- [ ] 在线容器无 GT；
- [ ] YOLO 故障不阻塞 VLA 控制链。

## 7. 成员 A / 组长：需要冻结和放行的事项

### 7.1 尚待确认或交付

1. 确认 Canonical Schema v1 的最终版本和 Owner；
2. 确认 Golden Episode 的目录、字段、动作旋转表示和夹爪物理含义；
3. 确认 Arm_A/Arm_B 是否分成两个 Episode，或一个完整 Episode 中如何明确分段；
4. 指定 C Recorder 的 PR 和验收负责人；
5. 审核 B GUI 键采完整 PR；
6. 审核 D/E/F 是否都用同一 Golden Episode 通过；
7. 只在 Replay、两个 Loader、QA 全部 PASS 后放行第一小批；
8. 小批验证通过后再批准扩大数量。

### 7.2 建议组长使用的放行 Gate

```markdown
- [ ] B：场景、三相机、7D 动作、GUI 键采、双臂 Linux 测试通过
- [ ] C：Recorder、Replay、Split、Manifest 和 Golden Episode 已交付
- [ ] D：Golden Arm_B Episode 可转换并被 RLDS Loader 完整遍历
- [ ] E：Golden Arm_A Episode 可转换并被 LeRobot/OpenPI Loader 完整遍历
- [ ] F：Golden Episode 通过 QA，GT 隔离和 Split 检查通过
- [ ] 所有产物记录 Git SHA、数据 SHA、场景 SHA 和复现命令
- [ ] 先采小批并抽查回放，不直接扩大规模
```

## 8. 成员 B：剩余工作与边界

### 8.1 已完成或已有证据

- 终端步进式键盘控制底层链路；
- 统一 7 维动作；
- Arm_A/Arm_B 控制适配；
- 三路 1280×720 RGB/CAS；
- Observation 校验、检查点和安全停止；
- Arm_A/Arm_B 终端 1 步与 10 步烟囱；
- Isaac Sim GUI 直接按键功能；
- Arm_A GUI 单步 PASS；
- 相关单元测试和本地完整测试。

### 8.2 当前仍需完成

1. Arm_A GUI 10 步 Linux 验证；
2. Arm_B GUI 1 步 Linux 验证；
3. Arm_B GUI 10 步 Linux 验证；
4. 整理三个结果 JSON、日志、短视频和 Git SHA；
5. 确认完整测试和代码格式检查通过；
6. 按组长规则只提交一次完整 GUI 键采 PR；
7. C 发布 Recorder API 后，将 Observation/Action/Event 接入 Recorder；
8. 完成至少 50 次 Reset、碰撞、可达性和固定 Seed 报告；
9. 后续补充脚本专家、故障注入和回放配合。

### 8.3 B 当前可以做与不能做的事情

可以做：

- GUI 操作演示；
- 1～10 步小规模 Smoke；
- 录屏；
- 验证按键、机械臂、相机、CAS、检查点和安全退出；
- 配合 C 定义 Recorder 接口。

暂时不能宣称：

- 已产生正式 Canonical Episode；
- 已产生可直接用于 π0.5/OpenVLA 训练的数据；
- 已满足大规模数据采集条件；
- Smoke 轨迹就是正式训练数据。

## 9. 队员之间是否依赖 B 的代码

### C 是否依赖 B

是。C 需要 B 提供：

- 新鲜 Observation；
- 三相机 ID、尺寸、图像路径/摘要；
- 机器人状态；
- 7 维 Action；
- 控制频率和渲染频率；
- 场景 SHA、控制器 Git SHA；
- FSM/事件和安全停止结果。

不过 C 可以先独立完成 Recorder API、Schema 校验、临时 Fixture、Replay 和 Split 工具，不必等 B 全部测试结束才开始。

### D/E 是否直接依赖 B

D/E 不应该直接读取 B 的 Smoke 文件格式，也不应该围绕 B 的临时 JSONL 写第二套转换器。正确依赖是：

```text
B 产生仿真 Observation/Action
    → C Recorder 写成 Canonical Episode
        → D 转 RLDS
        → E 转 LeRobot/OpenPI
```

所以 D/E 当前可以先完成转换器和合成测试，但真实端到端验收需要 C 的 Golden Episode。

### F 是否依赖 B

F 可以先完成 QA CLI、类别表、Exporter 和错误 Fixture；真实图像、离线 GT、Golden Episode QA 与回放证据需要 B+C 的产物。

## 10. 简短说明

> 我同步了 2026-08-03 的远端仓库，并按当前冻结合同整理了正式键采/规模化采集前的待交付物。B 目前已打通 GUI 键盘控制、三路 RGB/CAS 和双臂 Smoke，但这些结果仍明确标记为 `smoke_only`，不是正式 Canonical Episode。当前第一阻塞项是 C 的 Canonical Recorder、Replay、Split、Manifest 和 Golden Episode；D/E 需要用这条真实 Golden Episode 完成 RLDS/LeRobot Loader 验收，F 需要完成 Canonical QA、YOLO/COCO 和 GT 隔离检查。建议大家先分别确认本文件中自己的“验收标准”，由 A 冻结口径，全部通过后先放行小批量采集，不直接开始大规模采集。

## 11. 需要确认的事项

确认以下问题：

1. C 的 Recorder 函数/API 输入具体是什么；
2. Canonical v1 是否完全采用本文列出的目录和字段；
3. `tcp_pose` 四元数是否固定为 `xyzw`；
4. Action 是否固定为 `[dx,dy,dz,dax,day,daz,gripper]`，旋转为 rotation vector；
5. 夹爪开/合的端点和实际物理行为；
6. Arm_A 与 Arm_B Episode 的切分方法；
7. Golden Episode 由谁采、保存在哪里、由谁签字 PASS；
8. D/E/F 各自读取 Golden Episode 的命令；
9. 小批量数量和通过条件；
10. 哪些文件进入 GitHub，哪些大文件进入团队数据盘。

