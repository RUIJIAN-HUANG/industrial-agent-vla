# =============================================================================
# π0.5 模型 GPU 服务器部署 Dockerfile
# 负责人：E（pi0.5 / openpi）
# 方案书出处：§3.3 / §3.3.1 openpi-service 部署在独立 Ubuntu 22.04 + JAX + uv 环境
#             不能与角色 D 的 PyTorch 环境混用，通过 Docker 实现环境隔离
# 构建产物：openpi-service 容器（WebSocket 推理服务）
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 基础镜像：NVIDIA CUDA 12.4.1 + cuDNN + Ubuntu 22.04
#    - 方案书明确要求 Ubuntu 22.04
#    - π0.5 推理依赖 JAX CUDA 后端，必须基于官方 CUDA 镜像
#    - CUDA 12.4 与 jax[cuda12] 轮子 ABI 匹配，避免运行时报错
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# -----------------------------------------------------------------------------
# 2. 系统依赖
#    - git / git-lfs: clone openpi 仓库与处理大文件指针
#    - python3.11 / python3.11-venv / python3-pip: uv 安装基础与项目运行时
#    - ffmpeg: openpi 数据预处理 / 视频解码
#    - libgl1 / libglib2.0-0: cv2 (opencv-python) 运行时依赖
#    - ca-certificates / curl: 拉取远端资源与 https 校验
#    - 清理 apt 缓存以减小镜像体积
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 把 python3.11 设为默认 python，保证 uv 与 openpi 使用一致的 3.11 解释器
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# -----------------------------------------------------------------------------
# 3. 安装 uv（openpi 官方依赖管理器）
#    官方步骤：pip install uv
#    uv 用于 uv sync 锁定 openpi 依赖并创建虚拟环境
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

# -----------------------------------------------------------------------------
# 4. clone openpi 仓库（含子模块）
#    - --recurse-submodules: 拉取 openpi 依赖的子模块
#    - GIT_LFS_SKIP_SMUDGE=1: clone 时不拉 LFS 大文件（权重/数据走 volume，不进镜像）
#    镜像只保留代码与依赖定义，不打包模型权重与训练数据
# -----------------------------------------------------------------------------
ENV GIT_LFS_SKIP_SMUDGE=1
RUN git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git /workspace/openpi

WORKDIR /workspace/openpi

# -----------------------------------------------------------------------------
# 5. 安装 openpi 依赖
#    官方步骤：GIT_LFS_SKIP_SMUDGE=1 uv sync 然后 GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
#    - uv sync: 按 openpi 的 pyproject.toml 创建 .venv 并安装全部依赖
#    - uv pip install -e .: 以可编辑模式安装 openpi 本体，便于后续 patch
#    GIT_LFS_SKIP_SMUDGE=1 已在 ENV 设置，避免安装过程触发 LFS 拉取
# -----------------------------------------------------------------------------
RUN uv sync \
    && uv pip install -e .

# -----------------------------------------------------------------------------
# 6. 安装 GPU 版 JAX
#    基础镜像已带 CUDA 12.4 + cuDNN，仅需 jax[cuda12] 轮子
#    覆盖 openpi 默认安装的 CPU 版 jax（若 pyproject.toml 默认拉的是 CPU 版）
# -----------------------------------------------------------------------------
RUN uv pip install --upgrade "jax[cuda12]"

# -----------------------------------------------------------------------------
# 7. 复制本项目代码（角色 E 交付物）
#    - services/pi05/src/: 含 pi05.py、openpi_service.py 等核心代码
#    - scripts/: pi05 工具脚本（如 convert_openpi.py）
#    - configs/: 服务配置
#    注意：data/ 不 COPY，数据通过 volume 挂载到 /workspace/project/data
#          权重不 COPY，通过 volume 挂载到 /workspace/openpi/checkpoints
# -----------------------------------------------------------------------------
COPY services/pi05/src/ /workspace/project/services/pi05/src/
COPY scripts/ /workspace/project/scripts/
COPY configs/ /workspace/project/configs/

# 启动工作目录设为项目根，便于 `python -m services.pi05.src.openpi_service` 解析模块
WORKDIR /workspace/project

# 让 Python 能找到项目根下的包
ENV PYTHONPATH=/workspace/project:${PYTHONPATH}

# -----------------------------------------------------------------------------
# 7.5 安装本项目服务依赖（openpi venv 不含 FastAPI/uvicorn/msgpack）
#    方案书 §7.1：openpi_service 需要 FastAPI + uvicorn 提供 WebSocket 与 HTTP 端点；
#    msgpack 用于高性能序列化（与 openpi 官方兼容，fallback JSON）。
#    numpy 已由 openpi 依赖树安装，此处不重复安装以避免版本冲突。
# -----------------------------------------------------------------------------
RUN /workspace/openpi/.venv/bin/pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    msgpack

# -----------------------------------------------------------------------------
# 8. 运行时环境变量
#    PI05_SERVICE_MODE=dummy:
#        默认 dummy 模式，不 import openpi，便于联调与健康检查
#        生产推理通过 `docker run -e PI05_SERVICE_MODE=real` 切换
#    PI05_SERVICE_HOST=0.0.0.0:
#        容器内必须监听 0.0.0.0，否则宿主无法访问映射端口
#    PI05_SERVICE_PORT=8101:
#        HTTP 主路径（POST /v1/infer、GET /health、POST /v1/cancel）走 8101，
#        与冻结架构文档中的 π0.5 服务端口对齐。
#    PI05_CHECKPOINT_SHA / PI05_NORM_STATS_SHA:
#        权重与归一化统计的不可变摘要（方案书 §4 / 契约 §6.2）。
#        默认值为 64 个 0 的占位 sha256，仅用于 dummy 联调；
#        生产运行时必须通过 `docker run -e` 注入真实 sha256:<64hex>。
#    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9:
#        JAX 启动时预占 90% 显存，避免碎片化与多次分配开销
#    OPENPI_DATA_HOME=/workspace/openpi_data:
#        openpi 数据/缓存主目录，指向 volume，避免写入镜像层
# -----------------------------------------------------------------------------
ENV PI05_SERVICE_MODE=dummy \
    PI05_SERVICE_HOST=0.0.0.0 \
    PI05_SERVICE_PORT=8101 \
    PI05_CHECKPOINT_SHA=sha256:0000000000000000000000000000000000000000000000000000000000000000 \
    PI05_NORM_STATS_SHA=sha256:0000000000000000000000000000000000000000000000000000000000000000 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    OPENPI_DATA_HOME=/workspace/openpi_data

# -----------------------------------------------------------------------------
# 9. 健康检查（方案书 §7.1：/health 端点）
#    每 30s 探测一次，超时 3s，连续失败 3 次标记 unhealthy
#    探测 HTTP 主路径 GET /health（与契约 §6 一致）
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8101/health || exit 1

# -----------------------------------------------------------------------------
# 10. 暴露服务端口
#    HTTP 主路径（POST /v1/infer、GET /health、POST /v1/cancel）走 8101
# -----------------------------------------------------------------------------
EXPOSE 8101

# -----------------------------------------------------------------------------
# 11. 声明 volume（运行时挂载，绝不打包进镜像）
#     /workspace/openpi_data:        openpi 数据/缓存（OPENPI_DATA_HOME）
#     /workspace/project/data:       本项目数据集 / 录制样本
#     /workspace/openpi/checkpoints: 模型权重（base 权重与微调后权重）
#     -> 权重与数据均通过 `docker run -v` 挂载，镜像可复用、可迁移
# -----------------------------------------------------------------------------
VOLUME ["/workspace/openpi_data", "/workspace/project/data", "/workspace/openpi/checkpoints"]

# -----------------------------------------------------------------------------
# 12. 启动命令：openpi_service HTTP 服务
#     契约 §2 / §6 / §7 / §8：HTTP 主路径 POST /v1/infer、GET /health、POST /v1/cancel
#     通过 uvicorn CLI 显式指定 --port 8101，覆盖源码默认端口；
#     使用 openpi venv 的 python -m uvicorn，确保使用已安装 uvicorn 的解释器。
#     默认 dummy 模式启动；真实推理时 `docker run -e PI05_SERVICE_MODE=real`
# -----------------------------------------------------------------------------
CMD ["/workspace/openpi/.venv/bin/python", "-m", "uvicorn", "services.pi05.src.openpi_service:app", "--host", "0.0.0.0", "--port", "8101"]
