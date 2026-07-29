FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG OPENVLA_OFT_COMMIT=e4287e94541f459edc4feabc4e181f537cd569a8
LABEL org.opencontainers.image.source="https://github.com/RUIJIAN-HUANG/industrial-agent-vla"
LABEL org.opencontainers.image.openvla-oft-revision="${OPENVLA_OFT_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OPENVLA_OFT_UPSTREAM_DIR=/opt/openvla-oft \
    OPENVLA_OFT_CHECKPOINT_DIR=/models/openvla-oft \
    INDUSTRIAL_AGENT_CAS_ROOT=/var/lib/industrial-agent/cas \
    OPENVLA_OFT_USE_MOCK=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/moojink/openvla-oft.git /opt/openvla-oft \
    && git -C /opt/openvla-oft checkout --detach "${OPENVLA_OFT_COMMIT}" \
    && test "$(git -C /opt/openvla-oft rev-parse HEAD)" = "${OPENVLA_OFT_COMMIT}"

WORKDIR /opt/openvla-oft
RUN python -m pip install --upgrade "pip==24.3.1" \
    && python -m pip install -e .

WORKDIR /workspace/project
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY services/__init__.py ./services/__init__.py
COPY services/openvla_oft/ ./services/openvla_oft/

RUN python -m pip install -e . \
    && python -m pip install -e services/openvla_oft

EXPOSE 8102
VOLUME ["/models/openvla-oft", "/var/lib/industrial-agent/cas"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8102/health', timeout=3)"

CMD ["python", "services/openvla_oft/scripts/run_service.py", "--host", "0.0.0.0", "--port", "8102"]
