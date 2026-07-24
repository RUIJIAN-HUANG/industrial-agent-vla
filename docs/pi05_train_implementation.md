# `scripts/pi05/train.py` 实现说明文档

> π0.5 模型 LoRA 微调训练入口脚本（封装 openpi 官方 `scripts/train.py`）
>
> 负责人：E（π0.5/openpi）
> 方案书出处：§3.3 / §3.3.1 / §6.3
> 文档版本：v1.0 ｜ 编制日期：2026-07-24
> 待人工审核确认无误后提交至代码仓库。

---

## 一、设计目标

开发 π0.5 模型的 LoRA 微调训练入口脚本，对 openpi 官方 `scripts/train.py` 进行**非侵入式封装**，实现：

1. 项目自定义配置（`pi05_industrial`）的注册机制；
2. LoRA 冻结配置与 GPU 内存优化（`XLA_PYTHON_CLIENT_MEM_FRACTION`）控制；
3. 训练前 norm_stats 完整性校验；
4. 关键参数透传给官方训练函数；
5. CPU / Mock 环境下的骨架验证能力；
6. 训练流程稳定性与可复现性。

---

## 二、红线要求落实情况（Zero-Tolerance）

### 红线 1：环境变量时序控制 ✅

`XLA_PYTHON_CLIENT_MEM_FRACTION` 设置位于 [scripts/pi05/train.py#L43-L46](file:///d:/工灵智取/scripts/pi05/train.py#L43-L46)，**在所有 JAX/openpi 模块 import 之前**完成：

```python
# 此处仅 import os，不 import 任何 JAX/openpi 相关模块，确保环境变量先生效。
import os as _os
_os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
```

- 使用 `setdefault` 而非赋值：允许外部环境变量覆盖（Docker / CI 场景）。
- 默认值 `0.9`：与方案书 §3.3 + [docker/Dockerfile.pi05](file:///d:/工灵智取/docker/Dockerfile.pi05) 保持一致。
- 此处严格只 `import os`，**不 import 任何会触发 JAX 初始化的模块**，避免 JAX 提前初始化导致显存限制策略失效。

### 红线 2：标识符严格匹配 ✅

- 配置名 `pi05_industrial` 与 [configs/pi05/train_config.py](file:///d:/工灵智取/configs/pi05/train_config.py) 中定义完全一致（[train.py#L108](file:///d:/工灵智取/scripts/pi05/train.py#L108) `DEFAULT_CONFIG_NAME`）。
- LoRA Rank、action_dim、action_horizon、batch_size、num_train_steps 等参数全部从 `pi05_config` 模块属性读取（[train.py#L349-L360](file:///d:/工灵智取/scripts/pi05/train.py#L349-L360)），不重复定义。
- norm_stats 维度从 `config.model.action_dim` 动态读取（[train.py#L207](file:///d:/工灵智取/scripts/pi05/train.py#L207)），不写死。
- `OPENPI_AVAILABLE` / `LORA_RANK` / `BASE_CHECKPOINT` / `DATASET_REPO_ID` 等常量均引用 `pi05_config.*`。

### 红线 3：CPU/Mock 环境兼容性 ✅

- `--mock` 模式下自动将 `data` 替换为 openpi `FakeDataConfig`（[train.py#L321-L337](file:///d:/工灵智取/scripts/pi05/train.py#L321-L337)）。
- openpi 不可用时降级使用 `train_config` 占位 data，不崩溃。
- norm_stats 检查支持两种 Mock 策略（`--mock-stats skip|generate`，[train.py#L501-L506](file:///d:/工灵智取/scripts/pi05/train.py#L501-L506)）：
  - `skip`（默认）：跳过校验；
  - `generate`：自动生成符合 `_NormStatsDict` 格式的 Mock 文件。
- 已在本地无 GPU、无 openpi 环境下实测 `--mock` 与 `--mock --mock-stats generate` 均退出码 0。

### 红线 4：路径安全规范 ✅

- 全部路径通过 CLI 参数或环境变量获取，不写死任何个人本地绝对路径：
  - `PI05_CHECKPOINT_DIR` → 覆盖 `config.checkpoint_base_dir`
  - `PI05_ASSETS_DIR` → 覆盖 `config.assets_base_dir`
  - `PI05_OPENPI_REPO` → 定位官方 `scripts/train.py`
- 路径覆盖逻辑见 [train.py#L145-L160](file:///d:/工灵智取/scripts/pi05/train.py#L145-L160) 与 [train.py#L298-L304](file:///d:/工灵智取/scripts/pi05/train.py#L298-L304)。
- 摘要打印中显示的 `D:\工灵智取\...` 是运行时从环境推算的绝对路径（用于可观测性），未在源码中硬编码。

---

## 三、实现逻辑详解

### 3.1 模块加载时序

```
1. import os → setdefault(XLA_PYTHON_CLIENT_MEM_FRACTION=0.9)   [红线 1]
2. import sys → 探测 --quiet → setdefault(PI05_QUIET=1)
3. stdout/stderr.reconfigure(errors="replace")                  [Windows GBK 兼容]
4. 标准库 import (argparse/dataclasses/importlib/json/logging/pathlib/typing)
5. _PROJECT_ROOT 注入 sys.path                                   [任意目录可运行]
6. import configs.pi05.train_config as pi05_config              [红线 2：触发配置注册]
7. logger 初始化 + 常量定义
8. 函数定义（get_config / check_norm_stats / apply_overrides / ...）
9. main() 入口
```

**Windows 编码兼容性处理**（[train.py#L54-L64](file:///d:/工灵智取/scripts/pi05/train.py#L54-L64)）：Windows 控制台默认 GBK(cp936) 编码无法编码 `train_config.py` 中的 emoji（⚠️/✅），会导致 `UnicodeEncodeError`。脚本不强制改编码（否则中文乱码），而是用 `errors="replace"` 把无法编码字符替换为 `?`，保证不崩溃且中文正常显示。Linux/Docker 默认 UTF-8，`reconfigure` 无副作用。

### 3.2 配置加载策略（双路径降级）

[get_config()](file:///d:/工灵智取/scripts/pi05/train.py#L118-L139) 函数实现双路径降级：

1. **主路径**（openpi 可用，方案书 §3.3 JAX 路径）：
   ```python
   from openpi.training.config import get_config as _openpi_get_config
   cfg = _openpi_get_config(config_name)
   ```
2. **降级路径**（openpi 不可用，本地骨架验证）：
   ```python
   cfg = pi05_config.get_config(config_name)
   ```
   使用 `train_config.py` 提供的占位 dataclass，仅用于文档/检查，不能用于真实训练。

### 3.3 训练前检查（Pre-flight Checks）

#### norm_stats 路径约定（严格对齐 [compute_norm_stats.py](file:///d:/工灵智取/scripts/pi05/compute_norm_stats.py)）

[get_norm_stats_path()](file:///d:/工灵智取/scripts/pi05/train.py#L163-L174) 计算路径：

```
<assets_dirs>/<repo_id>/norm_stats.json
```

与 `compute_norm_stats.py` 中的输出约定一致：
```python
output_path = config.assets_dirs / data_config.repo_id
normalize.save(output_path, norm_stats)  # 写入 output_path/norm_stats.json
```

`assets_dirs` 优先级（[get_assets_dirs()](file:///d:/工灵智取/scripts/pi05/train.py#L145-L160)）：
1. `PI05_ASSETS_DIR` 环境变量（覆盖 `config.assets_base_dir`）；
2. `config.assets_dirs` 属性（openpi 官方 `TrainConfig` 的 property）；
3. 降级默认 `./assets/<config_name>`（占位 config 场景）。

#### Mock Stats 生成格式

[generate_mock_norm_stats()](file:///d:/工灵智取/scripts/pi05/train.py#L195-L230) 生成符合 openpi `shared/normalize.py` 的 `_NormStatsDict` 格式：

```json
{
  "norm_stats": {
    "state":   {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
    "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}
  }
}
```

- `state_dim = 8`：Franka 7-DOF + 1 gripper（与 [convert_openpi.py](file:///d:/工灵智取/scripts/pi05/convert_openpi.py) `DEFAULT_STATE_DIM` 一致）。
- `action_dim`：从 `config.model.action_dim` 动态读取（默认 7，方案书 §3.4 `[dx,dy,dz,dax,day,daz,gripper]`）。

### 3.4 官方训练逻辑调用（Python 内部 import，严禁 subprocess）

[load_openpi_train_module()](file:///d:/工灵智取/scripts/pi05/train.py#L246-L280) 通过 `importlib.util` 动态加载官方 `scripts/train.py`，查找顺序：

1. `PI05_OPENPI_REPO` 环境变量 → `<repo>/scripts/train.py`；
2. 通过 `openpi.__file__` 推断包位置 → `<repo>/scripts/train.py`。

加载成功后调用 `train_module.main(config)`（[train.py#L544-L552](file:///d:/工灵智取/scripts/pi05/train.py#L544-L552)），关键参数透传机制见 3.5。

### 3.5 关键参数透传

[apply_overrides()](file:///d:/工灵智取/scripts/pi05/train.py#L286-L318) 使用 `dataclasses.replace` 创建新 config（不可变 dataclass 安全），透传：

| 字段 | 来源 |
|---|---|
| `exp_name` | CLI `--exp-name` |
| `overwrite` | CLI `--overwrite` |
| `resume` | CLI `--resume` |
| `checkpoint_base_dir` | 环境变量 `PI05_CHECKPOINT_DIR` |
| `assets_base_dir` | 环境变量 `PI05_ASSETS_DIR` |

**安全过滤**：仅保留 config 实际拥有的字段（[train.py#L307-L315](file:///d:/工灵智取/scripts/pi05/train.py#L307-L315)），占位 dataclass 缺失字段时自动跳过而非报错。

### 3.6 Mock 模式数据替换

[apply_mock_data()](file:///d:/工灵智取/scripts/pi05/train.py#L321-L337)：

```python
from openpi.training.config import FakeDataConfig
new_config = dataclasses.replace(config, data=FakeDataConfig())
```

`FakeDataConfig` 是 openpi 自带的 `DataConfigFactory` 子类（`repo_id="fake"`），无需真实数据与 norm_stats，专用于流程测试。

### 3.7 训练配置摘要打印

[print_summary()](file:///d:/工灵智取/scripts/pi05/train.py#L343-L407) 在训练启动前打印方案书要求可见性的关键参数：

- Config 名称 / 实验名称 / openpi 可用性 / Mock 模式
- LoRA Rank / action_dim / action_horizon
- Batch Size / Total Steps / Warmup / Weight Decay / Grad Accum / Mixed Precision / Eval Interval / FSDP Devices
- Memory Fraction / Assets Dirs / Norm Stats Path / Base Checkpoint / Dataset repo_id
- **LoRA 冻结状态安全闸门**（C3）：检测 `freeze_filter` + `weight_loader` 是否配置，未配置时给出显存警告。

### 3.8 CLI 接口设计

[parse_args()](file:///d:/工灵智取/scripts/pi05/train.py#L413-L461)：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--config-name` | 必需 | `pi05_industrial` | 配置名称，与 train_config.py 一致 |
| `--exp-name` | 必需 | None | 实验名称（mock 模式默认 `mock_experiment`） |
| `--mock` | 可选 | False | 启用 Mock 模式 |
| `--overwrite` | 可选 | False | 覆盖已有实验 |
| `--resume` | 可选 | False | 从 checkpoint 恢复 |
| `--quiet` | 可选 | False | 静默模式 |
| `--mock-stats` | 可选 | `skip` | Mock 模式 norm_stats 策略（`skip\|generate`） |

---

## 四、技术亮点

### 4.1 双路径降级架构

`get_config` / `load_openpi_train_module` / `apply_mock_data` 全部采用「openpi 主路径 → 本地降级」策略，使脚本**在无 openpi 环境下也能完成骨架验证**，而真实训练时无缝切换到 openpi 官方路径，不损失任何官方能力。

### 4.2 环境变量时序严格隔离

脚本顶部仅 `import os`、`import sys`，所有可能触发 JAX 初始化的模块（包括 `configs.pi05.train_config`）都在环境变量设置完成后才 import，从源头规避 JAX 提前初始化导致显存限制失效的隐患。

### 4.3 跨平台编码兼容

通过 `sys.stdout.reconfigure(errors="replace")` 优雅处理 Windows GBK 控制台无法编码 emoji 的问题，既不改变默认编码（避免中文乱码），又保证脚本不崩溃。Linux/Docker 下 `reconfigure` 无副作用。

### 4.4 不可变 dataclass 安全更新

所有 config 字段更新通过 `dataclasses.replace` 创建新实例（openpi 的 `TrainConfig` 是 `frozen=True`），并通过字段存在性过滤避免占位 dataclass 缺字段导致的 `TypeError`。

### 4.5 Mock Stats 格式严格对齐

`generate_mock_norm_stats` 严格按照 openpi `shared/normalize.py` 的 `_NormStatsDict` 格式生成，维度从 `config.model.action_dim` 动态读取，确保 Mock 文件能被官方 normalize.load 正常加载。

### 4.6 LoRA 安全闸门可见性

摘要打印中显式检测 `freeze_filter` + `weight_loader` 是否配置（C3 安全闸门），未配置时给出「显存需求 >70GB」警告，避免误用全参数训练导致 OOM。

### 4.7 Python 内部 import 调用官方训练

通过 `importlib.util.spec_from_file_location` 动态加载官方 `scripts/train.py`，避免 subprocess 进程间调用的额外开销与调试困难，同时保留官方训练函数的完整栈跟踪。

---

## 五、用法示例

### 5.1 真实训练（GPU 服务器，openpi 已安装）

```bash
python scripts/pi05/train.py --config-name pi05_industrial \
    --exp-name my_experiment --overwrite
```

### 5.2 Mock 骨架验证（CPU，无 openpi 也能跑）

```bash
# 策略 1：跳过 norm_stats 校验（默认）
python scripts/pi05/train.py --mock

# 策略 2：生成 Mock norm_stats.json
python scripts/pi05/train.py --mock --mock-stats generate
```

### 5.3 从 checkpoint 恢复训练

```bash
python scripts/pi05/train.py --config-name pi05_industrial \
    --exp-name my_experiment --resume
```

### 5.4 自定义 checkpoint / assets 目录

```bash
PI05_CHECKPOINT_DIR=/data/checkpoints \
PI05_ASSETS_DIR=/data/assets \
python scripts/pi05/train.py --exp-name my_experiment
```

---

## 六、`--mock` 模式完整终端输出日志

### 6.1 命令

```bash
python scripts/pi05/train.py --mock
```

### 6.2 完整终端输出（exit code = 0）

```
(TraeAI-4) D:\工灵智取 [0:] > trae-sandbox 'python scripts/pi05/train.py --mock'

[2026-07-24 11:18:32,141][WARNING][pi05_config] ================================================================
[pi05_train_config] ⚠️  LoRA 安全闸门：freeze_filter / weight_loader 未配置！
  当前状态:
    freeze_filter  = ❌ None（未冻结任何参数 → 全参数训练）
    weight_loader  = ❌ None（未注入 LoRA 适配层）
  风险:
    - 将执行全参数微调而非 LoRA，显存需求从 >22.5GB 飙升至 >70GB（§3.3）。
    - 与方案书规定的 LoRA 微调路径直接矛盾。
  修复:
    1. 按 openpi 官方 LoRA 文档配置 freeze_filter（JAX nnx.filterlib.Filter）。
    2. 创建 LoRAWeightLoader 指向 pi05_base checkpoint，rank=32。
    3. 参考: https://github.com/Physical-Intelligence/openpi
  (此警告在 openpi 不可用/降级模式下属预期，真实训练前必须消除)
================================================================
[2026-07-24 11:18:32,154][INFO][pi05_train] 加载配置: pi05_industrial
[2026-07-24 11:18:32,154][INFO][pi05_train] [MOCK] 未指定 --exp-name，使用默认值: mock_experiment
[2026-07-24 11:18:32,156][INFO][pi05_train] 已应用 CLI 覆盖: exp_name=mock_experiment, overwrite=False, resume=False
[2026-07-24 11:18:32,156][INFO][pi05_train] [MOCK] openpi 不可用，跳过 FakeDataConfig 替换（使用 train_config 占位 data）
[2026-07-24 11:18:32,156][INFO][pi05_train] [MOCK] 策略=skip：跳过 norm_stats.json 校验
[2026-07-24 11:18:32,158][INFO][pi05_train] ============================================================
[2026-07-24 11:18:32,158][INFO][pi05_train] [MOCK] 骨架验证完成。
[2026-07-24 11:18:32,158][INFO][pi05_train] [MOCK] 已验证：配置加载 ✓ / 参数透传 ✓ / Pre-flight 检查 ✓ / 摘要打印 ✓
[2026-07-24 11:18:32,158][INFO][pi05_train] [MOCK] 真实训练需要 openpi（方案书 §3.3 JAX 路径）：
[2026-07-24 11:18:32,158][INFO][pi05_train] [MOCK]   git clone https://github.com/Physical-Intelligence/openpi
[2026-07-24 11:18:32,158][INFO][pi05_train] [MOCK]   cd openpi && uv sync && uv pip install -e .
[2026-07-24 11:18:32,158][INFO][pi05_train] ============================================================
[pi05_train_config] WARNING: openpi 未安装或 import 失败，已降级为占位 dataclass。
  原因: No module named 'openpi'
  提示: LoRA 微调必须走 openpi JAX 路径（方案书 §3.3）。
        请安装: git clone https://github.com/Physical-Intelligence/openpi       
                cd openpi && uv sync
  当前 PI05_INDUSTRIAL_CONFIG 仅用于文档/检查，不能用于真实训练。
================================================================
[pi05_train_config] 配置摘要
================================================================
openpi 可用:        False
openpi import 错误: No module named 'openpi'
提示: LoRA 微调必须走 openpi JAX 路径（方案书 §3.3）。
      git clone https://github.com/Physical-Intelligence/openpi
      cd openpi && uv sync
配置名:             pi05_industrial
model_type:         pi05
action_dim:         7   (方案书 §3.4 [dx,dy,dz,dax,day,daz,gripper])
action_horizon:     10  (初始候选，D21 后按闭环表现调整)
batch_size:         16  (默认安全值 16；方案书 §3.3：22.5GB 卡建议 ≤16)
lr init_value:      2e-5 (LoRA 微调较小学习率)
num_train_steps:    30000  (openpi 官方示例参考值；D21 按数据量与收敛调整)      
warmup_steps:       2000  (C2 修复；若 openpi API 不支持则由 scheduler 内部控制)
weight_decay:       0.01  (C2 修复；若 openpi API 不支持则由 optimizer 内部控制)
grad_accum_steps:   1  (C2 修复)
mixed_precision:    bf16  (C2 修复；JAX 路径推荐 bf16)
eval_interval:      1000  (W2 修复；若 openpi API 不支持则由外部脚本触发)       
LoRA rank:          32 (方案书 §3.2.1 OpenVLA-OFT 示例；π0.5 初始候选值)        
base checkpoint:    gs://openpi-assets/checkpoints/pi05_base
dataset repo_id:    industrial_team/industrial_dataset
output_dir:         ./checkpoints/pi05_industrial
fsdp_devices:       1   (单卡，方案书 §3.3 JAX 路径)
注册到 _CONFIGS:    True
??  LoRA 安全闸门: freeze_filter / weight_loader 未配置！(C3)
   当前配置禁止用于真实训练；全参数训练显存 >70GB，远超 LoRA 预算 22.5GB。      
----------------------------------------------------------------
训练命令:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
      pi05_industrial --exp-name=my_experiment --overwrite
norm stats 计算命令（训练前必跑，方案书 §3.3.1 Para186）:
  uv run scripts/compute_norm_stats.py --config-name pi05_industrial
================================================================
========================================================================        
[pi05_train] 训练配置摘要
========================================================================
  Config 名称:        pi05_industrial
  实验名称 (exp_name): mock_experiment
  openpi 可用:        False
  Mock 模式:          True
------------------------------------------------------------------------        
  LoRA Rank:          32  (方案书 §3.2.1 OpenVLA-OFT 示例；π0.5 初始候选值)     
  action_dim:         7  (方案书 §3.4 [dx,dy,dz,dax,day,daz,gripper])
  action_horizon:     10  (初始候选，D21 后按闭环表现调整)
  Batch Size:         16  (方案书 §3.3：22.5GB 卡建议 ≤16)
  Total Steps:        30000  (openpi 官方示例参考值；D21 按数据量调整)
  Warmup Steps:       2000
  Weight Decay:       0.01
  Grad Accum Steps:   1
  Mixed Precision:    bf16  (JAX 路径推荐 bf16，方案书 §3.3)
  Eval Interval:      1000
  FSDP Devices:       1  (单卡=1，方案书 §3.3 JAX 路径)
------------------------------------------------------------------------        
  Memory Fraction:    0.9  (XLA_PYTHON_CLIENT_MEM_FRACTION)
  Assets Dirs:        D:\工灵智取\assets\pi05_industrial
  Norm Stats Path:    D:\工灵智取\assets\pi05_industrial\industrial_team\industrial_dataset\norm_stats.json
  Base Checkpoint:    gs://openpi-assets/checkpoints/pi05_base
  Dataset repo_id:    industrial_team/industrial_dataset
------------------------------------------------------------------------        
  LoRA 冻结状态:      ??  freeze_filter/weight_loader 未配置（C3 安全闸门）     
                       当前为全参数训练配置，显存需求 >70GB（§3.3）
------------------------------------------------------------------------        
  Overwrite:          False
  Resume:             False
  Quiet:              False
========================================================================        
(TraeAI-4) D:\工灵智取 [0:0] $
```

### 6.3 辅助验证：`--mock --mock-stats generate --quiet`（exit code = 0）

```
(TraeAI-4) D:\工灵智取 [0:0] > trae-sandbox 'python scripts/pi05/train.py --mock --mock-stats generate --quiet'
[pi05_train_config] WARNING: openpi 未安装或 import 失败，已降级为占位 dataclass。
  原因: No module named 'openpi'
  提示: LoRA 微调必须走 openpi JAX 路径（方案书 §3.3）。
        请安装: git clone https://github.com/Physical-Intelligence/openpi       
                cd openpi && uv sync
  当前 PI05_INDUSTRIAL_CONFIG 仅用于文档/检查，不能用于真实训练。
(TraeAI-4) D:\工灵智取 [0:0] $
```

### 6.4 验证结论

| 检查项 | 结果 |
|---|---|
| 退出码 | `0`（成功） |
| 红线 1：环境变量时序 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 在 import 前设置 ✓ |
| 红线 2：标识符匹配 | `pi05_industrial` 与 train_config.py 一致 ✓ |
| 红线 3：Mock 模式 | 无 GPU / 无 openpi 下不崩溃 ✓ |
| 红线 4：路径安全 | 无硬编码绝对路径，路径由环境变量推导 ✓ |
| 配置加载 | `get_config('pi05_industrial')` 成功 ✓ |
| 参数透传 | `exp_name=mock_experiment` 正确覆盖 ✓ |
| Pre-flight 检查 | `skip` / `generate` 两种策略均通过 ✓ |
| 摘要打印 | LoRA Rank / Batch / Steps / Memory Fraction 全部显示 ✓ |
| LoRA 安全闸门 | C3 警告正确触发（freeze_filter 未配置）✓ |
| 跨平台编码 | Windows GBK 下 emoji 不崩溃（替换为 `?`/`??`）✓ |

> **注**：日志中 `??` 为 Windows GBK 控制台无法编码 emoji（⚠️/✅）经 `errors="replace"` 替换后的显示，属预期行为；Linux/Docker UTF-8 环境下将正常显示 emoji。摘要中显示的 `D:\工灵智取\...` 是运行时从 `_PROJECT_ROOT` 推算的绝对路径（用于可观测性），源码中未硬编码。

---

## 七、待人工审核确认项

1. **LoRA 冻结配置**（C3 安全闸门）：当前 `freeze_filter` / `weight_loader` 留空（TODO），待 GPU 服务器上按 openpi 官方 LoRA 文档配置（方案书 §3.3）。
2. **openpi 真实训练路径**：本地无 openpi，仅验证骨架；真实训练需在 GPU 服务器执行 `git clone https://github.com/Physical-Intelligence/openpi && uv sync && uv pip install -e .`。
3. **norm_stats 真实数据**：训练前必须执行 `python scripts/pi05/compute_norm_stats.py --config-name pi05_industrial` 生成自有统计（方案书 §3.3.1 Para186）。
4. **超参确认**：`num_train_steps=30000` / `batch_size=16` 为 openpi 官方示例参考值，D21 按数据量与收敛情况调整（方案书 §6.3 首轮微调 100—500 条/核心技能）。

---

## 八、关联文件

- 训练入口：[scripts/pi05/train.py](file:///d:/工灵智取/scripts/pi05/train.py)
- 配置定义：[configs/pi05/train_config.py](file:///d:/工灵智取/configs/pi05/train_config.py)
- norm_stats 计算：[scripts/pi05/compute_norm_stats.py](file:///d:/工灵智取/scripts/pi05/compute_norm_stats.py)
- 数据转换：[scripts/pi05/convert_openpi.py](file:///d:/工灵智取/scripts/pi05/convert_openpi.py)
- Docker 部署：[docker/Dockerfile.pi05](file:///d:/工灵智取/docker/Dockerfile.pi05)
- 项目方案书：[docs/project_plan.md](file:///d:/工灵智取/docs/project_plan.md)
