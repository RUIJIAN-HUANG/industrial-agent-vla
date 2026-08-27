# 三模型生产部署

本目录只部署 π0.5、OpenVLA-OFT 和 YOLO 三个模型服务。Isaac Sim 与
Supervisor 必须在安装了 Isaac Sim 5.1 的目标机进程中运行，不进入这份普通模型
Compose。

## 拓扑

```text
Isaac 目标机（Isaac Sim + Supervisor，写 CAS）
                 │ HTTP + 同一个共享 CAS
                 ▼
Linux GPU 模型机（三个 Docker 容器，只读 CAS）
  π0.5 :8101     OpenVLA-OFT :8102     YOLO :8103
```

如果两台机器不能挂载同一个 NFS/SMB 目录，当前 `cas://sha256/...` 合同无法跨机
解析；必须先增加远程 CAS 服务，不能只把本地路径字符串发到另一台机器。

## 生产约束

- Compose 只使用已经构建、测试并推送的镜像 digest，不在生产机临时构建；
- 三个服务强制使用 real 模式，没有 Mock 自动回退；
- checkpoint、norm stats 和 YOLO 类别表必须提供完整非零 SHA-256；
- 模型和 CAS 以只读方式挂载，只有独立缓存目录可写；
- Compose 不包含 Supervisor，也不接受 offline GT、Stage Transform 或硬编码状态；
- `MODEL_BIND_IP=127.0.0.1` 只允许模型机本机访问。远程 Isaac 目标机需要填写模型
  机的受控内网地址，并同时配置主机防火墙；不要把未加密 HTTP 端口暴露到公网。

## 1. 准备环境文件

在 Linux GPU 模型机上执行：

```bash
cp deploy/.env.production.example deploy/.env.production
chmod 600 deploy/.env.production
```

填写真实值。所有 `REPLACE_WITH_*` 都是故意设置的失败哨兵，不能用于生产。
`.env.production` 已被 Git 忽略，禁止提交个人路径、内部地址或凭据。

主机必须提前安装 Docker Engine、Docker Compose v2、NVIDIA 驱动和 NVIDIA
Container Toolkit。三个 `*_CACHE_DIR_HOST` 目录需要对容器运行用户可写；模型目录与
`SHARED_CAS_DIR` 只需可读。

## 2. 启动前校验资产

```bash
python deploy/preflight.py \
  --env-file deploy/.env.production \
  --phase assets \
  --output artifacts/deployment/model-assets-preflight.json
```

预检会实际读取文件并复核：

- π0.5 完整 checkpoint 目录摘要与 norm-stats 文件摘要；
- OpenVLA checkpoint manifest、manifest 中每个文件、norm stats 和动作合同；
- YOLO 权重文件摘要；
- 三个不可变镜像 digest、端口、GPU ID、共享 CAS 和缓存挂载。

任何一项失败都禁止执行 `docker compose up`。

## 3. 检查 Compose 展开结果

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.models.production.yaml \
  config --quiet
```

正式运行：

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.models.production.yaml \
  pull

docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.models.production.yaml \
  up -d --wait
```

## 4. 启动后校验服务身份

```bash
python deploy/preflight.py \
  --env-file deploy/.env.production \
  --phase services \
  --output artifacts/deployment/model-services-preflight.json
```

这一阶段不只检查 HTTP 200，还会确认：

- `status=ready`；
- 服务名分别为 `pi05`、`openvla_oft`、`yolo`；
- OpenVLA 和 YOLO 不是 mock 模式；
- `/health` 返回的 checkpoint、norm-stats、class-map 和 YOLO config SHA 与环境
  文件完全一致。

模型机本地还可以查看：

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.models.production.yaml \
  ps
```

## 5. Isaac 目标机生产配置

将 `configs/agent.default.json` 复制为目标机专用的
`configs/agent.production.json`，至少替换：

- `perception.base_url` 为模型机 `http://<内网地址>:8103`；
- `executors.pi05.base_url` 为 `http://<内网地址>:8101`；
- `executors.openvla_oft.base_url` 为 `http://<内网地址>:8102`；
- 所有 `REPLACE_WITH_PINNED_SHA`；
- `image_cas.root` 为 Isaac 目标机看到的同一共享 CAS；
- `isaac_runtime.command_ledger_path` 和证据输出目录；
- 必要时提供目标机可解析的 Franka USD 路径。

先在 Isaac 目标机运行无动作验收：

```bash
<ISAAC_PYTHON> simulation/run_production_runtime_acceptance.py \
  --config configs/agent.production.json \
  --output artifacts/isaac-runtime/production-runtime-acceptance.json
```

只有报告为 `PASS` 后，才允许启动包含机械臂动作的 Supervisor 主流程。

## 停止和回滚

停止服务：

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.models.production.yaml \
  down
```

回滚时修改三个 `*_IMAGE_DIGEST` 和对应模型摘要，重新运行完整的 assets/services
两阶段预检。禁止用 `latest` 或只修改镜像 tag 的方式回滚。
