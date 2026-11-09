from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.pipeline import (
    StreamingTTSPipeline,
    StreamMetricsSummary,
    simulate_llm_token_stream,
)
from app.tts_engine import ResilientTTSEngine, pcm_to_wav_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

tts_engine = ResilientTTSEngine()
tts_pipeline = StreamingTTSPipeline(engine=tts_engine, max_queue_size=4)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm conversational filler cache on startup.
    try:
        warmed = await tts_engine.warm_up_fillers(languages=("en", "de"))
        logger.info("Pre-warmed %d English & German conversational fillers.", warmed)
    except Exception as exc:
        logger.warning("Filler warm-up skipped on startup: %s", exc)
    yield


app = FastAPI(
    title="VoiceStream-Eval: Real-Time Streaming TTS & Voice Benchmark API",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="LLM response text to synthesize")
    language: str = Field(default="en", pattern="^(en|de)$")
    mode: str = Field(default="streaming", pattern="^(streaming|batch)$")
    token_delay_ms: float = Field(default=35.0, ge=0.0, le=250.0)
    simulate_slow_primary_sec: float = Field(default=0.0, ge=0.0, le=10.0)
    simulate_primary_error: bool = Field(default=False)


@app.get("/", response_class=HTMLResponse)
async def index_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>VoiceStream-Eval TTS Service is running.</h1>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "supported_languages": ["en", "de"],
        "cached_fillers_in_memory": len(tts_engine._memory_cache),
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint exposing TTFA, RTF, Barge-In, and Failover histograms."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/synthesize")
async def synthesize_http(req: SynthesizeRequest):
    # Synthesize text using either Streaming Clause Chunking or Batch Mode
    # and return telemetry metrics along with base64 WAV audio.
    metrics = StreamMetricsSummary(mode=req.mode, language=req.language)
    token_stream = simulate_llm_token_stream(req.text, token_delay_ms=req.token_delay_ms)

    pcm_parts: list[bytes] = []
    sample_rate = tts_engine.sample_rate

    if req.mode == "batch":
        stream_iter = tts_pipeline.synthesize_batch_from_tokens(
            token_stream,
            language=req.language,
            metrics_out=metrics,
        )
    else:
        stream_iter = tts_pipeline.stream_from_tokens(
            token_stream,
            language=req.language,
            metrics_out=metrics,
            simulate_slow_primary_sec=req.simulate_slow_primary_sec,
            simulate_primary_error=req.simulate_primary_error,
        )

    async for chunk in stream_iter:
        pcm_parts.append(chunk.pcm_bytes)
        sample_rate = chunk.sample_rate

    full_pcm = b"".join(pcm_parts)
    wav_bytes = pcm_to_wav_bytes(full_pcm, sample_rate=sample_rate) if full_pcm else b""

    return JSONResponse(
        {
            "mode": metrics.mode,
            "language": metrics.language,
            "ttfa_ms": round(metrics.ttfa_ms, 2),
            "total_turn_time_ms": round(metrics.total_turn_time_ms, 2),
            "total_compute_time_ms": round(metrics.total_compute_time_ms, 2),
            "total_audio_duration_ms": round(metrics.total_audio_duration_ms, 2),
            "rtf": round(metrics.rtf, 4),
            "clauses_emitted": metrics.clauses_emitted,
            "chunks_emitted": metrics.chunks_emitted,
            "engines_used": metrics.engines_used,
            "wav_base64": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else "",
        }
    )


@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    # Bidirectional WebSocket for real-time voice synthesis and instant caller barge-in:
    # - Client sends: {"type": "synthesize", "text": "...", "language": "en"|"de", "mode": "streaming"|"batch"}
    # - Client sends: {"type": "barge_in"} at any time to cancel active synthesis within milliseconds.

    await websocket.accept()
    cancel_event: Optional[asyncio.Event] = None
    active_task: Optional[asyncio.Task] = None

    async def _run_synthesis_session(payload: dict, session_cancel: asyncio.Event):
        text = str(payload.get("text", "")).strip()
        language = str(payload.get("language", "en")).lower()[:2]
        mode = str(payload.get("mode", "streaming"))
        token_delay_ms = float(payload.get("token_delay_ms", 35.0))
        simulate_slow_sec = float(payload.get("simulate_slow_primary_sec", 0.0))

        metrics = StreamMetricsSummary(mode=mode, language=language)
        token_stream = simulate_llm_token_stream(
            text, token_delay_ms=token_delay_ms, cancel_event=session_cancel
        )

        if mode == "batch":
            stream_iter = tts_pipeline.synthesize_batch_from_tokens(
                token_stream,
                language=language,
                cancel_event=session_cancel,
                metrics_out=metrics,
            )
        else:
            stream_iter = tts_pipeline.stream_from_tokens(
                token_stream,
                language=language,
                cancel_event=session_cancel,
                metrics_out=metrics,
                simulate_slow_primary_sec=simulate_slow_sec,
            )

        clause_buffers: dict[int, list[bytes]] = {}
        clause_meta: dict[int, dict] = {}

        async for chunk in stream_iter:
            idx = chunk.clause_index
            clause_buffers.setdefault(idx, []).append(chunk.pcm_bytes)
            clause_meta[idx] = {
                "clause_text": chunk.clause_text,
                "clause_index": idx,
                "engine_used": chunk.engine_used,
                "sample_rate": chunk.sample_rate,
            }
            if chunk.is_final_chunk:
                clause_pcm = b"".join(clause_buffers.pop(idx, []))
                wav_bytes = pcm_to_wav_bytes(clause_pcm, sample_rate=chunk.sample_rate)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "clause_audio",
                            "clause_index": idx,
                            "clause_text": chunk.clause_text,
                            "engine_used": chunk.engine_used,
                            "ttfa_ms": round(metrics.ttfa_ms, 2),
                            "rtf": round(metrics.rtf, 4),
                            "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                        }
                    )
                )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "turn_complete",
                    "mode": metrics.mode,
                    "language": metrics.language,
                    "ttfa_ms": round(metrics.ttfa_ms, 2),
                    "total_turn_time_ms": round(metrics.total_turn_time_ms, 2),
                    "rtf": round(metrics.rtf, 4),
                    "clauses_emitted": metrics.clauses_emitted,
                    "engines_used": metrics.engines_used,
                    "barge_in_triggered": metrics.barge_in_triggered,
                    "barge_in_cancel_latency_ms": round(metrics.barge_in_cancel_latency_ms, 3),
                }
            )
        )

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "barge_in":
                if cancel_event is not None:
                    cancel_event.set()
                await websocket.send_text(json.dumps({"type": "barge_in_ack"}))

            elif msg_type == "synthesize":
                if cancel_event is not None:
                    cancel_event.set()
                if active_task is not None and not active_task.done():
                    with contextlib.suppress(Exception):
                        await active_task

                cancel_event = asyncio.Event()
                active_task = asyncio.create_task(_run_synthesis_session(data, cancel_event))

    except WebSocketDisconnect:
        if cancel_event is not None:
            cancel_event.set()
        if active_task is not None and not active_task.done():
            active_task.cancel()

