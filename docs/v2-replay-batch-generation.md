# V2 回放轨迹批处理生成

`simulation/generate_v2_replay_batch.py` 从一条成功的 Canonical V2 母 Episode
生成确定性的 `diverse_low` 与 `approach_curve` 回放配置。当前冻结任务
`P01_TO_S11` 和 `W01_TO_S14` 共用同一入口；任务 ID 与训练指令从母 Episode
读取，不能由命令行改写。

生成目录包含：

- `configs/*.json`：逐轨迹配置、种子、变体、抬升量、末端 Y/Z 偏移和计划动作哈希；
- `manifest.json` 与 `manifest.sha256`：母 Episode/场景/配置/轨迹/结果哈希；
- `commands.ps1`：按计划逐条运行并最终验收的确切命令。

计划阶段会拒绝失败的母 Episode、错误的任务/指令组合、场景 SHA 不一致、与母轨迹
相同或批内重复的动作轨迹。`finalize` 阶段会拒绝缺失、失败、配置被改写、任务或场景
身份不符、实际动作哈希偏离计划以及重复的输出；被拒绝项保留作 QA 证据，但不会标记
为训练就绪。

## 训练就绪批处理的确切顺序

以下 PowerShell 序列在仓库根目录执行。`<...>` 必须替换为本机的绝对路径；
`<SOURCE_EPISODE>` 可为任务一 P01 或任务二 W01 的成功母 Episode。

```powershell
$FrozenSha = (git rev-parse HEAD).Trim()
$IsaacPython = '<ISAAC_SIM_PYTHON.BAT_ABSOLUTE_PATH>'
$OpenPiRoot = '<CLEAN_OPENPI_ROOT_ABSOLUTE_PATH>'
$BatchRoot = '<BATCH_ROOT_ABSOLUTE_PATH>'
$CanonicalRoot = '<CANONICAL_V2_ROOT_ABSOLUTE_PATH>'
$CasRoot = '<CAS_ROOT_ABSOLUTE_PATH>'

python simulation\generate_v2_replay_batch.py plan `
  --source-episode '<SOURCE_EPISODE>' `
  --output-dir "$BatchRoot\plan" `
  --episode-root $CanonicalRoot `
  --cas-root $CasRoot `
  --artifact-root "$BatchRoot\artifacts" `
  --scene-output-root "$BatchRoot\scenes" `
  --scene-config simulation\configs\single_bin_scene_v2.json `
  --split train `
  --base-seed 1000 `
  --diverse-low-count 3 `
  --approach-curve-count 4 `
  --python-command $IsaacPython `
  --frozen-collection-sha $FrozenSha `
  --openpi-root $OpenPiRoot

& "$BatchRoot\plan\commands.ps1"

python scripts\pi05\convert_openpi_v2.py `
  --data-dir $CanonicalRoot `
  --split-registry '<SHA_VERIFIED_SPLIT_REGISTRY_JSON>' `
  --preflight-only

python scripts\pi05\convert_openpi_v2.py `
  --data-dir $CanonicalRoot `
  --split-registry '<SHA_VERIFIED_SPLIT_REGISTRY_JSON>' `
  --output-dir '<LEROBOT_DATASET_ROOT>' `
  --repo-id '<ORG/REPO_ID>'
```

`commands.ps1` 已逐项展开相同的场景配置、母 Episode、任务身份、seed、profile、
variant 和 offset，最后自动执行 `finalize`。任一 Isaac 命令或最终验收返回非零时，
PowerShell 因 `$ErrorActionPreference = 'Stop'` 立即停止。只有最终清单同时满足
`status="ACCEPTED"`、`training_ready=true`、`rejected=0` 才能进入后续转换。

默认 3 条 `diverse_low` 使用确定性的 `0/-2/+2 mm` Y 偏移和 seed 决定的
`0.2/0.3/0.5 mm` 平滑抬升；4 条 `approach_curve` 分别使用冻结变体 1–4。
若任务二母轨迹在首次闭合夹爪前少于 6 个动作，`approach_curve` 会明确拒绝，必须
重新采集合格母轨迹，不能退化为无差异副本。
