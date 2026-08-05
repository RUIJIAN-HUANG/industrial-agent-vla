FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.source="https://github.com/RUIJIAN-HUANG/industrial-agent-vla"
LABEL org.opencontainers.image.ultralytics-version="8.4.104"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    INDUSTRIAL_AGENT_CAS_ROOT=/var/lib/industrial-agent/cas \
    YOLO_CHECKPOINT_PATH=/models/yolo/model.pt \
    YOLO_DEVICE=0 \
    YOLO_USE_MOCK=0

WORKDIR /workspace/project
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY services/__init__.py ./services/__init__.py
COPY services/yolo/ ./services/yolo/

RUN python -m pip install --upgrade "pip==25.1.1" \
    && python -m pip install . "services/yolo[model]"

EXPOSE 8103
VOLUME ["/models/yolo", "/var/lib/industrial-agent/cas"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8103/health', timeout=3)"

CMD ["yolo-service", "--host", "0.0.0.0", "--port", "8103"]
