from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = BASE_DIR / "vendor"
LOWMEMORY_SCRIPT = VENDOR_DIR / "evaluate_jsonl_lowmemory.py"
PYTHON_BIN = os.environ.get("METRICS_API_PYTHON", "python")
SERVICE_GPU_DEVICE = os.environ.get("SERVICE_GPU_DEVICE", "cuda:0")
SERVICE_FACTSPOTTER_DEVICE = os.environ.get("SERVICE_FACTSPOTTER_DEVICE", "cuda")
DEFAULT_ALIGN_CKPT = os.environ.get("ALIGN_CKPT_PATH", "/models/AlignScore-large.ckpt")
DEFAULT_BLEURT_CKPT = os.environ.get("BLEURT_CHECKPOINT", "/models/BLEURT-20")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "256"))

LOG_LEVEL = os.environ.get("METRICS_API_LOG_LEVEL", "INFO").upper()
MAX_JOB_HISTORY = int(os.environ.get("METRICS_API_JOB_HISTORY", "100"))
JOB_TIMEOUT_SEC = int(os.environ.get("METRICS_API_JOB_TIMEOUT_SEC", "300"))
MAX_LOG_TAIL_CHARS = int(os.environ.get("METRICS_API_LOG_TAIL_CHARS", "20000"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("metrics-api")

_jobs: deque[dict] = deque(maxlen=MAX_JOB_HISTORY)
_jobs_by_id: dict[str, dict] = {}

app = FastAPI(title="Metrics API", version="1.0.0", default_response_class=ORJSONResponse)
_eval_lock = asyncio.Lock()
_state = {"busy": False, "current_job": None, "started_at": None}


class EvalRequest(BaseModel):
    records: List[dict] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=lambda: ["bleu", "bertscore", "bleurt", "alignscore", "factspotter"])
    bertscore_lang: str = "en"
    align_ckpt_path: str = DEFAULT_ALIGN_CKPT
    bleurt_checkpoint: str = DEFAULT_BLEURT_CKPT
    align_model: str = "roberta-large"
    align_batch_size: int = 16
    align_eval_mode: str = "nli_sp"
    factspotter_model_name: str = "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
    factspotter_batch_size: int = 16
    factspotter_entailment_threshold: float = 0.5
    infolm_model_name: str = "bert-base-uncased"
    infolm_information_measure: str = "kl_divergence"
    infolm_temperature: float = 0.25
    infolm_batch_size: int = 16
    infolm_use_idf: bool = True


class EvalResponse(BaseModel):
    request_id: str
    duration_sec: float
    metrics: dict
    stdout: str = ""
    stderr: str = ""


class HealthResponse(BaseModel):
    ok: bool
    busy: bool
    current_job: Optional[str] = None
    started_at: Optional[float] = None
    visible_devices: Optional[str] = None
    cuda_visible_devices: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_cmd(cmd: List[str]) -> str:
    # Keep command readable, but avoid accidentally logging very long paths/args
    return shlex.join(cmd)


def _record_job(job: dict) -> None:
    _jobs.appendleft(job)
    _jobs_by_id[job["request_id"]] = job


def _update_job(request_id: str, **updates) -> None:
    job = _jobs_by_id.get(request_id)
    if not job:
        return
    job.update(updates)


def _normalize_metrics(metrics: List[str]) -> List[str]:
    allowed = {"bleu", "bertscore", "bleurt", "infolm", "alignscore", "factspotter"}
    normalized = []
    seen = set()
    for m in metrics:
        key = m.strip().lower()
        if key not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported metric: {m}")
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    if not normalized:
        raise HTTPException(status_code=400, detail="At least one metric must be requested")
    return normalized


def _build_cmd(input_path: Path, output_path: Path, req: EvalRequest) -> List[str]:
    requested = set(_normalize_metrics(req.metrics))
    flags = []
    for metric in ["bleu", "bertscore", "bleurt", "infolm", "alignscore", "factspotter"]:
        if metric not in requested:
            flags.append(f"--no-{metric}")

    cmd = [
        PYTHON_BIN,
        str(LOWMEMORY_SCRIPT),
        str(input_path),
        "--output-path",
        str(output_path),
        "--bertscore-lang",
        req.bertscore_lang,
        "--align-ckpt-path",
        req.align_ckpt_path,
        "--align-model",
        req.align_model,
        "--align-batch-size",
        str(req.align_batch_size),
        "--align-device",
        SERVICE_GPU_DEVICE,
        "--align-eval-mode",
        req.align_eval_mode,
        "--factspotter-model-name",
        req.factspotter_model_name,
        "--factspotter-device",
        SERVICE_FACTSPOTTER_DEVICE,
        "--factspotter-batch-size",
        str(req.factspotter_batch_size),
        "--factspotter-entailment-threshold",
        str(req.factspotter_entailment_threshold),
        "--bleurt-checkpoint",
        req.bleurt_checkpoint,
        "--infolm-model-name",
        req.infolm_model_name,
        "--infolm-information-measure",
        req.infolm_information_measure,
        "--infolm-temperature",
        str(req.infolm_temperature),
        "--infolm-batch-size",
        str(req.infolm_batch_size),
    ]
    if not req.infolm_use_idf:
        cmd.append("--infolm-no-idf")
    cmd.extend(flags)
    return cmd


def _append_job_output(request_id: str, stream_name: str, text: str) -> None:
    job = _jobs_by_id.get(request_id)
    if not job:
        return

    key = f"{stream_name}_tail"
    previous = job.get(key) or ""
    job[key] = (previous + text)[-MAX_LOG_TAIL_CHARS:]


def _run_subprocess(cmd: List[str], request_id: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", "/cache/huggingface")
    env.setdefault("TRANSFORMERS_CACHE", "/cache/huggingface")
    # Enforce self-contained/offline model resolution for evaluation workers.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env.setdefault("NLTK_DATA", "/cache/nltk_data")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("PYTHONUNBUFFERED", "1")

    vendor_path = str(VENDOR_DIR)
    base_path = str(BASE_DIR)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join([p for p in [vendor_path, base_path, existing] if p])

    logger.info("job=%s subprocess starting: %s", request_id, shlex.join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    _update_job(request_id, pid=proc.pid)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def pump(stream, stream_name: str, parts: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                parts.append(line)
                _append_job_output(request_id, stream_name, line)

                if stream_name == "stdout":
                    logger.info("job=%s stdout: %s", request_id, line.rstrip())
                else:
                    logger.warning("job=%s stderr: %s", request_id, line.rstrip())
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=pump,
        args=(proc.stdout, "stdout", stdout_parts),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=(proc.stderr, "stderr", stderr_parts),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        returncode = proc.wait(timeout=JOB_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        logger.error("job=%s timed out after %s sec; killing process group", request_id, JOB_TIMEOUT_SEC)

        _update_job(
            request_id,
            status="failed",
            error=f"Job timed out after {JOB_TIMEOUT_SEC} seconds",
        )

        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

        returncode = proc.returncode if proc.returncode is not None else -9

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)

    logger.info("job=%s subprocess finished returncode=%s", request_id, returncode)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


async def _evaluate_from_path(input_path: Path, req: EvalRequest) -> EvalResponse:
    request_id = str(uuid.uuid4())
    queued_at = time.time()

    job = {
        "request_id": request_id,
        "status": "queued",
        "pid": None,
        "queued_at": queued_at,
        "queued_at_iso": _now_iso(),
        "started_at": None,
        "started_at_iso": None,
        "finished_at": None,
        "finished_at_iso": None,
        "duration_sec": None,
        "metrics": _normalize_metrics(req.metrics),
        "input_path": str(input_path),
        "command": None,
        "returncode": None,
        "error": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    _record_job(job)

    logger.info("job=%s queued metrics=%s input=%s", request_id, job["metrics"], input_path)

    async with _eval_lock:
        started = time.time()

        _state["busy"] = True
        _state["current_job"] = request_id
        _state["started_at"] = started

        _update_job(
            request_id,
            status="running",
            started_at=started,
            started_at_iso=_now_iso(),
        )

        logger.info(
            "job=%s started after_queue_sec=%.3f",
            request_id,
            started - queued_at,
        )

        try:
            with tempfile.TemporaryDirectory(prefix="metrics-api-") as td:
                output_path = Path(td) / "metrics.json"
                cmd = _build_cmd(input_path=input_path, output_path=output_path, req=req)

                _update_job(request_id, command=_redact_cmd(cmd))

                proc = await asyncio.to_thread(_run_subprocess, cmd, request_id)

                _update_job(
                    request_id,
                    returncode=proc.returncode,
                    stdout_tail=proc.stdout[-4000:],
                    stderr_tail=proc.stderr[-4000:],
                )

                if proc.returncode != 0:
                    error = {
                        "message": "Metric evaluation failed",
                        "request_id": request_id,
                        "command": shlex.join(cmd),
                        "stdout": proc.stdout[-4000:],
                        "stderr": proc.stderr[-4000:],
                        "returncode": proc.returncode,
                    }

                    _update_job(
                        request_id,
                        status="failed",
                        finished_at=time.time(),
                        finished_at_iso=_now_iso(),
                        duration_sec=round(time.time() - started, 3),
                        error=error,
                    )

                    logger.error("job=%s failed returncode=%s", request_id, proc.returncode)

                    raise HTTPException(status_code=500, detail=error)

                if not output_path.exists():
                    error = "Metrics file was not produced"

                    _update_job(
                        request_id,
                        status="failed",
                        finished_at=time.time(),
                        finished_at_iso=_now_iso(),
                        duration_sec=round(time.time() - started, 3),
                        error=error,
                    )

                    logger.error("job=%s failed: %s", request_id, error)

                    raise HTTPException(status_code=500, detail=error)

                metrics = json.loads(output_path.read_text(encoding="utf-8"))
                duration = round(time.time() - started, 3)

                _update_job(
                    request_id,
                    status="succeeded",
                    finished_at=time.time(),
                    finished_at_iso=_now_iso(),
                    duration_sec=duration,
                )

                logger.info("job=%s succeeded duration_sec=%.3f", request_id, duration)

                return EvalResponse(
                    request_id=request_id,
                    duration_sec=duration,
                    metrics=metrics,
                    stdout=proc.stdout[-4000:],
                    stderr=proc.stderr[-4000:],
                )

        finally:
            _state["busy"] = False
            _state["current_job"] = None
            _state["started_at"] = None


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        ok=True,
        busy=bool(_state["busy"]),
        current_job=_state["current_job"],
        started_at=_state["started_at"],
        visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


@app.get("/v1/jobs")
def list_jobs() -> dict:
    return {
        "busy": bool(_state["busy"]),
        "current_job": _state["current_job"],
        "jobs": list(_jobs),
    }


@app.get("/v1/jobs/{request_id}")
def get_job(request_id: str) -> dict:
    job = _jobs_by_id.get(request_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/v1/metrics")
def metrics_options() -> dict:
    return {
        "supported_metrics": ["bleu", "bertscore", "bleurt", "infolm", "alignscore", "factspotter"],
        "default_align_ckpt_path": DEFAULT_ALIGN_CKPT,
        "default_bleurt_checkpoint": DEFAULT_BLEURT_CKPT,
        "single_request_concurrency": 1,
    }


@app.post("/v1/evaluate/records", response_model=EvalResponse)
async def evaluate_records(req: EvalRequest) -> EvalResponse:
    if not req.records:
        raise HTTPException(status_code=400, detail="records must be non-empty")
    _normalize_metrics(req.metrics)
    with tempfile.TemporaryDirectory(prefix="metrics-api-input-") as td:
        input_path = Path(td) / "input.jsonl"
        with input_path.open("w", encoding="utf-8") as f:
            for row in req.records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return await _evaluate_from_path(input_path, req)


@app.post("/v1/evaluate/file", response_model=EvalResponse)
async def evaluate_file(
    file: UploadFile = File(...),
    metrics: str = Form("bleu,bertscore,alignscore,factspotter"),
    bertscore_lang: str = Form("en"),
    align_ckpt_path: str = Form(DEFAULT_ALIGN_CKPT),
    bleurt_checkpoint: str = Form(DEFAULT_BLEURT_CKPT),
    align_model: str = Form("roberta-large"),
    align_batch_size: int = Form(16),
    align_eval_mode: str = Form("nli_sp"),
    factspotter_model_name: str = Form("Inria-CEDAR/FactSpotter-DeBERTaV3-Base"),
    factspotter_batch_size: int = Form(16),
    factspotter_entailment_threshold: float = Form(0.5),
    infolm_model_name: str = Form("bert-base-uncased"),
    infolm_information_measure: str = Form("kl_divergence"),
    infolm_temperature: float = Form(0.25),
    infolm_batch_size: int = Form(64),
    infolm_use_idf: bool = Form(True),
) -> EvalResponse:
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    req = EvalRequest(
        records=[],
        metrics=metric_list,
        bertscore_lang=bertscore_lang,
        align_ckpt_path=align_ckpt_path,
        bleurt_checkpoint=bleurt_checkpoint,
        align_model=align_model,
        align_batch_size=align_batch_size,
        align_eval_mode=align_eval_mode,
        factspotter_model_name=factspotter_model_name,
        factspotter_batch_size=factspotter_batch_size,
        factspotter_entailment_threshold=factspotter_entailment_threshold,
        infolm_model_name=infolm_model_name,
        infolm_information_measure=infolm_information_measure,
        infolm_temperature=infolm_temperature,
        infolm_batch_size=infolm_batch_size,
        infolm_use_idf=infolm_use_idf,
    )
    _normalize_metrics(req.metrics)

    suffix = Path(file.filename or "predictions.jsonl").suffix or ".jsonl"
    with tempfile.TemporaryDirectory(prefix="metrics-api-upload-") as td:
        input_path = Path(td) / f"upload{suffix}"
        size = 0
        with input_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large; max is {MAX_UPLOAD_MB} MB")
                out.write(chunk)
        await file.close()

        return await _evaluate_from_path(input_path, req)


@app.on_event("shutdown")
def _shutdown() -> None:
    cache_dir = Path("/tmp") / "metrics-api"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
