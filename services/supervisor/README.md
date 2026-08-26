# Supervisor 生产组合入口

本目录提供 V2 总控容器入口。入口只组合 π0.5/Arm_A 连续闭环，不加载模型或
Isaac Sim 权重，也不会在缺少平台环境时退回 Mock。V1 配置会在启动阶段被拒绝。

## 强制启动条件

1. Agent 配置中的 π0.5 checkpoint/norm-stats SHA 必须替换为完整的
   `sha256:<64 位十六进制>`；
2. π0.5 URL 必须可访问，且 `/health` 内容与配置一致；
3. 必须通过 `--environment-factory` 或
   `INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY` 注入真实平台环境；
4. Isaac Sim 环境工厂必须返回
   `industrial_agent.supervisor_main.EnvironmentHost`，并在 `run()` 期间持续泵送
   `IsaacMainThreadGate`；
5. 进程结束前必须取得确认的 `SafeStopReceipt`，否则退出码为 `3`。

工厂格式固定为：

```text
package.module:create_environment
```

工厂接收已经加载的 Agent 配置，返回 `ExecutionEnvironment` 或
`EnvironmentHost`。返回 `None`、缺少方法或返回 Mock 替代品都会在启动阶段失败。

## 本地入口

```bash
python -m industrial_agent.supervisor_main \
  --config /run/config/agent.production.json \
  --task /run/config/task.v2.p01-to-s11.json \
  --environment-factory simulation.production_runtime:create_host
```

仓库中的 `configs/agent.default.json` 含有故意保留的 SHA 占位符，不能直接作为生产
配置启动。示例任务位于 `configs/task.v2.p01-to-s11.example.json`。

## 构建容器

必须从仓库根目录构建，以便 Docker 能读取 `src/` 和 `pyproject.toml`：

```bash
docker build -f services/supervisor/Dockerfile -t industrial-agent-supervisor:local .
```

Supervisor 容器不应挂载模型权重、GT 或图像 CAS；图像以不可变 CAS URI 经过接口
传递。平台工厂属于 Isaac Adapter 的交付范围，未提供时入口明确拒绝启动。
