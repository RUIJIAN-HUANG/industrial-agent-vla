"""Multi-GPU HTTP gateway for the π0.5 service.

The OpenPI JAX policy is process-local.  A single policy process therefore
must not be advertised as using multiple GPUs just because several devices are
visible.  This launcher starts one normal ``openpi_service`` process per GPU
and keeps the public HTTP contract on one port.  Requests from the same
episode/task are consistently routed to the same worker; different keys are
spread across workers for inference throughput.

Environment variables:
    PI05_GPU_IDS: comma-separated physical GPU ids (for example ``0,1``).
                   ``PI05_GPU_ID`` remains a single-GPU compatibility fallback.
    PI05_SERVICE_PORT: public gateway port (default 8101).
    PI05_WORKER_PORT_BASE: first loopback worker port (default public+1).

The gateway intentionally proxies only the frozen HTTP endpoints.  Real mode
already rejects the legacy inline-pixel WebSocket transport.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool
except Exception as exc:  # pragma: no cover - exercised by image build, not unit tests
    raise RuntimeError("multi-GPU gateway requires fastapi and starlette") from exc


logger = logging.getLogger("pi05_multi_gpu")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][pi05_multi_gpu] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _parse_gpu_ids() -> list[str]:
    raw = os.environ.get("PI05_GPU_IDS") or os.environ.get("PI05_GPU_ID")
    if not raw:
        raise RuntimeError("PI05_GPU_IDS 未设置；请提供逗号分隔的 GPU id，例如 0,1")
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpu_ids:
        raise RuntimeError("PI05_GPU_IDS 不能为空")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RuntimeError(f"PI05_GPU_IDS 包含重复 GPU id: {raw!r}")
    return gpu_ids


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是正整数，当前值={raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} 必须是正整数，当前值={value}")
    return value


GPU_IDS = _parse_gpu_ids()
PUBLIC_PORT = _positive_int("PI05_SERVICE_PORT", 8101)
WORKER_PORT_BASE = _positive_int("PI05_WORKER_PORT_BASE", PUBLIC_PORT + 1)
if WORKER_PORT_BASE <= PUBLIC_PORT < WORKER_PORT_BASE + len(GPU_IDS):
    raise RuntimeError(
        "PI05_WORKER_PORT_BASE 必须与公共端口隔离，且至少能容纳所有 worker 端口"
    )


class Worker:
    def __init__(self, gpu_id: str, port: int) -> None:
        self.gpu_id = gpu_id
        self.port = port
        worker_env = os.environ.copy()
        worker_env["PI05_GPU_ID"] = gpu_id
        worker_env.pop("PI05_GPU_IDS", None)
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        worker_env["PI05_SERVICE_HOST"] = "127.0.0.1"
        worker_env["PI05_SERVICE_PORT"] = str(port)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "services.pi05.src.openpi_service:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=worker_env,
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


workers: list[Worker] = []
_route_lock = threading.Lock()


def _route_key(body: bytes, path: str) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return path
    if path.endswith("/infer"):
        return str(payload.get("episode_id") or payload.get("task_id") or path)
    return str(payload.get("task_id") or payload.get("episode_id") or path)


def _select_worker(body: bytes, path: str) -> Worker:
    if not workers:
        raise RuntimeError("没有可用的 π0.5 worker")
    key = _route_key(body, path).encode("utf-8")
    # Stable routing preserves each episode's executor-local state while still
    # distributing independent episodes across GPUs.
    index = int.from_bytes(hashlib.blake2s(key, digest_size=4).digest(), "big")
    return workers[index % len(workers)]


def _forward(
    worker: Worker,
    method: str,
    path: str,
    body: bytes | None = None,
) -> tuple[int, bytes, str]:
    request = UrlRequest(
        worker.url + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get_content_type()
    except URLError as exc:
        payload = json.dumps({"error": "worker_unavailable", "reason": str(exc)}).encode()
        return 503, payload, "application/json"


async def _proxy(request: Request, path: str) -> Response:
    body = await request.body()
    try:
        worker = _select_worker(body, path)
        status, payload, content_type = await run_in_threadpool(
            _forward, worker, request.method, path, body
        )
    except RuntimeError as exc:
        return JSONResponse(
            {"error": "worker_unavailable", "reason": str(exc)}, status_code=503
        )
    return Response(content=payload, status_code=status, media_type=content_type)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global workers
    workers = [
        Worker(gpu_id, WORKER_PORT_BASE + index)
        for index, gpu_id in enumerate(GPU_IDS)
    ]
    logger.info("π0.5 multi-GPU workers started: gpu_ids=%s", GPU_IDS)
    try:
        yield
    finally:
        for worker in workers:
            worker.stop()
        workers = []


app = FastAPI(title="π0.5 multi-GPU gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    checks = await asyncio.gather(
        *(
            run_in_threadpool(_forward, worker, "GET", "/health")
            for worker in workers
        ),
        return_exceptions=True,
    )
    payloads: list[dict[str, Any]] = []
    for worker, result in zip(workers, checks, strict=True):
        if isinstance(result, Exception):
            payloads.append({"gpu_id": worker.gpu_id, "status": "unavailable"})
            continue
        status, body, _content_type = result
        try:
            item = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            item = {"status": "unavailable", "reason": "invalid worker health response"}
        item["gpu_id"] = worker.gpu_id
        item["http_status"] = status
        payloads.append(item)

    all_ready = bool(payloads) and all(
        item.get("status") == "ready" and item.get("http_status") == 200
        for item in payloads
    )
    primary = dict(payloads[0]) if payloads else {"service": "pi05"}
    primary.pop("gpu_id", None)
    primary.pop("http_status", None)
    primary["status"] = "ready" if all_ready else "loading"
    primary["inference"] = {
        "strategy": "one-process-per-gpu",
        "gpu_ids": GPU_IDS,
        "workers": payloads,
    }
    return primary


@app.post("/v1/infer")
async def infer(request: Request) -> Response:
    return await _proxy(request, "/v1/infer")


@app.post("/v1/cancel")
async def cancel(request: Request) -> Response:
    return await _proxy(request, "/v1/cancel")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("PI05_SERVICE_HOST", "0.0.0.0"), port=PUBLIC_PORT)
