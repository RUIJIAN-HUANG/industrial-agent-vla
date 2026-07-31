FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/industrial-agent

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 supervisor \
    && mkdir -p /var/lib/industrial-agent \
    && chown supervisor:supervisor /var/lib/industrial-agent
USER supervisor
WORKDIR /var/lib/industrial-agent

ENTRYPOINT ["python", "-m", "industrial_agent.supervisor_main"]
