# 仓库目录与文件规范

本文件定义代码、配置、数据说明、模型服务、仿真、实验和比赛材料的唯一归属。
新成员上传文件前，应先按下表确定位置；不要在仓库根目录临时堆放文件。
当前冻结主线为 Supervisor、π0.5、OpenVLA-OFT、YOLO 四 Agent，以及双 Franka
固定串行协作。目录调整不得产生第二套生命周期、第二个同职责服务或其他仿真
平台主线。

## 1. 规范目录

```text
industrial-agent-vla/
├── .github/                    # CI、Issue/PR 模板、CODEOWNERS
├── configs/                    # 可提交的示例/默认配置，不含密钥和机器绝对路径
├── data/                       # 数据卡、清单、Schema、小型测试夹具；不存数据集
├── docs/
│   ├── official/              # 唯二官方 PDF 原件，不可修改
│   ├── assets/                # 冻结架构/分工图及文档小图
│   ├── source/                # 可修订方案的原始参考文件
│   ├── requirements/          # 官方指标追踪
│   ├── architecture/          # 架构和跨模块接口
│   └── project-management/    # WBS、排期、日报、风险和协作规则
├── experiments/               # 可复现实验定义和摘要，不存运行缓存
├── models/                    # 模型卡、下载说明、摘要，不存权重
├── reports/                   # 精选报告和证据索引，不存原始日志/长视频
├── schemas/                   # 跨进程机器可校验 JSON Schema
├── scripts/                   # 项目级维护、校验和演示脚本
├── services/
│   ├── openvla_oft/           # D：独立 OpenVLA-OFT 服务
│   ├── pi05/                  # E：独立 π0.5/openpi 服务
│   └── yolo/                  # F：同步调用、失败非门控的 YOLO 评分 sidecar
├── simulation/                # B：仿真环境、控制器、场景配置和环境适配
├── src/industrial_agent/      # A：轻量总 Agent 核心
└── tests/                     # 单元、契约、回归测试
```

未经 A 批准，不新增功能重复的一级目录。例如，不要同时创建 `model/`、
`model_code/` 和 `vla_models/`；模型服务统一进入 `services/`，模型元数据统一进入
`models/`。

## 2. 文件应该放在哪里

| 文件类型 | 唯一位置 | 可以进入 Git | 责任角色 |
|---|---|---:|---|
| 总 Agent Python 代码 | `src/industrial_agent/` | 是 | A |
| OpenVLA-OFT 服务代码 | `services/openvla_oft/` | 是 | D |
| π0.5/openpi 服务代码 | `services/pi05/` | 是 | E |
| YOLO Agent 服务代码 | `services/yolo/` | 是 | F |
| 仿真/机器人适配代码 | `simulation/` | 是 | B |
| 跨模块 JSON Schema | `schemas/` | 是 | A + 接口方 |
| 默认或示例配置 | `configs/` 或模块内 `configs/` | 是，不含私密值 | 模块负责人 |
| 小型、合成测试样本 | `data/fixtures/` | 是，必须可公开 | C/F |
| 数据集清单/数据卡 | `data/` | 是 | C/F |
| 原始/处理/生成数据 | 外部制品存储 | 否 | C |
| 模型卡/下载清单/SHA | `models/` | 是 | D/E/F |
| checkpoint/导出引擎 | 外部制品存储 | 否 | D/E |
| 实验配置与结果摘要 | `experiments/` | 是 | D/E/F |
| 原始日志、W&B/MLflow 目录 | 外部制品存储或本地忽略目录 | 否 | F |
| 测试代码 | `tests/` 或服务自身 `tests/` | 是 | 开发者 + F |
| 接口、架构、运行说明 | `docs/architecture/` | 是 | A + 接口方 |
| 报告与证据索引 | `reports/` | 是 | F |
| 官方 PDF | `docs/official/` | 是但不可修改 | A/F |

## 3. 命名规则

- 目录与代码文件：小写英文，使用 `snake_case`；服务目录可使用约定名
  `openvla_oft`、`pi05`。
- Markdown 文档：小写英文 `kebab-case.md`；冻结官方原件和已有每日任务文件除外。
- Python 测试：`test_<被测主题>.py`，测试函数为 `test_<行为>_<预期>`。
- 配置：`<模块>.<环境>.json|yaml`，例如 `agent.default.json`、
  `openvla.sim.yaml`；真实密钥通过环境变量或密钥管理系统提供。
- 实验：`YYYYMMDD_<模型>_<任务>_<短标识>/`；提交配置、Commit SHA、数据/权重
  SHA 和摘要，不提交完整输出。
- 报告和证据：名称必须能关联 Gate、Issue 或实验 ID，例如
  `G3_openvla_20-seed-summary.md`。

文件名不得包含个人姓名、`final-final`、`新建文件夹`、`副本` 等不可追踪描述。
同一逻辑文件只有一个权威版本；历史版本由 Git 保存，不用 `_v2_copy` 复制。

## 4. 模块最小结构

新增可运行模块或服务时至少包含：

```text
<module>/
├── README.md             # 负责人、能力、边界、启动命令、依赖和状态
├── pyproject.toml        # 或明确的依赖清单
├── src/                  # 生产代码
├── tests/                # 最小单元/契约测试
└── configs/              # 只提交 example/default
```

涉及跨进程通信的模块还必须：

1. 实现 `docs/architecture/interface-contracts.md` 中对应端点；
2. 使用 `schemas/` 中的合同版本；
3. 在 CI 中运行契约测试；
4. 在 README 说明 checkpoint、norm stats、坐标系和动作语义；
5. 不把 PyTorch/CUDA 与 JAX/openpi 依赖装进总 Agent 核心环境。

## 5. 配置与制品规则

- Git 中的配置必须可公开、可复现，使用相对路径或环境变量，不写
  `C:\Users\...`、`/home/<name>/...` 等个人路径。
- 正式实验必须记录：Git Commit、配置文件、随机种子、数据清单 SHA、权重 SHA、
  运行环境和结果索引。
- 权重、数据集、录像、机器人录包、引擎和原始日志放外部制品存储；Git 仅保存
  下载说明、SHA-256、许可证/来源和复现命令。
- 需要 Git LFS 时必须先由 A 确认配额、比赛提交方式和备份策略，再通过独立 PR
  修改 `.gitattributes`。

## 6. 上传前检查

```powershell
python scripts/check_repository_hygiene.py
python scripts/verify_official_baselines.py
python scripts/verify_project_frozen_inputs.py
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
git status
git diff --check
```

`check_repository_hygiene.py` 会拒绝已被 Git 跟踪的权重、数据录包、视频、缓存、
密钥文件以及大于 10 MiB 的普通文件。确有例外时，不得绕过脚本；应创建 Issue，
说明原因、许可证、大小、下载影响和替代方案，由 A/F 评审。
