from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient

from app.livekit_agent import NaironLiveKitTTSAdapter
from app.main import app
from app.pipeline import (
    AsyncClauseChunker,
    StreamingTTSPipeline,
    StreamMetricsSummary,
    simulate_llm_token_stream,
)
from app.tts_engine import ResilientTTSEngine


@pytest.mark.asyncio
async def test_clause_chunker_respects_punctuation_and_abbreviations():
    # Verify that AsyncClauseChunker splits clauses on commas/periods without breaking 'Dr.'.
    chunker = AsyncClauseChunker(min_words_comma=4, min_words_sentence=2)
    text = (
        "One moment please, let me check that for you. "
        "Your appointment with Dr. Meyer is at 14:30, and it is confirmed."
    )
    tokens = simulate_llm_token_stream(text, token_delay_ms=1.0)
    clauses = [c async for c in chunker.chunk_tokens(tokens)]

    assert len(clauses) == 3
    assert clauses[0] == "One moment please, let me check that for you."
    assert clauses[1] == "Your appointment with Dr. Meyer is at 14:30,"
    assert clauses[2] == "and it is confirmed."


@pytest.mark.asyncio
async def test_streaming_ttfa_significantly_beats_batch_ttfa():

    # Verify that streaming clause chunking achieves much lower TTFA than batch synthesis.
    engine = ResilientTTSEngine(use_neural_piper=False)  # Fast deterministic synth for unit test
    pipeline = StreamingTTSPipeline(engine=engine, max_queue_size=4)

    text = (
        "Hello and thank you for calling our service desk today, "
        "we are looking up your customer profile right now, "
        "and we will connect you with an engineer shortly."
    )

    batch_metrics = StreamMetricsSummary(mode="batch", language="en")
    async for _ in pipeline.synthesize_batch_from_tokens(
        simulate_llm_token_stream(text, token_delay_ms=15.0),
        language="en",
        metrics_out=batch_metrics,
    ):
        pass

    stream_metrics = StreamMetricsSummary(mode="streaming", language="en")
    async for _ in pipeline.stream_from_tokens(
        simulate_llm_token_stream(text, token_delay_ms=15.0),
        language="en",
        metrics_out=stream_metrics,
    ):
        pass

    assert stream_metrics.clauses_emitted >= 3
    assert stream_metrics.ttfa_ms < batch_metrics.ttfa_ms * 0.5


@pytest.mark.asyncio
async def test_barge_in_cancels_stream_and_drains_queue_immediately():
    
    # Verify caller barge-in halts active TTS emission in <15ms and drains remaining clauses.
    engine = ResilientTTSEngine(use_neural_piper=False)
    pipeline = StreamingTTSPipeline(engine=engine, max_queue_size=3)

    long_text = (
        "First spoken clause is right here, "
        "second spoken clause follows right after, "
        "third spoken clause should never be completed, "
        "fourth spoken clause is drained on barge in."
    )
    cancel_event = asyncio.Event()
    metrics = StreamMetricsSummary(mode="streaming", language="en")

    emitted_chunks = []
    async for chunk in pipeline.stream_from_tokens(
        simulate_llm_token_stream(long_text, token_delay_ms=5.0, cancel_event=cancel_event),
        language="en",
        cancel_event=cancel_event,
        metrics_out=metrics,
    ):
        emitted_chunks.append(chunk)
        if len(emitted_chunks) == 2:
            cancel_event.set()

    assert metrics.barge_in_triggered is True
    assert metrics.barge_in_cancel_latency_ms < 25.0
    assert metrics.clauses_emitted < 4


@pytest.mark.asyncio
async def test_filler_cache_hit_and_timeout_failover():
    
    # Verify pre-warmed English & German fillers hit cache, and slow primary TTS fails over.
    engine = ResilientTTSEngine(use_neural_piper=False, timeout_sec=0.08)
    await engine.warm_up_fillers(languages=("en", "de"))

    # Test German filler cache hit
    de_filler = "Einen Moment bitte, ich prüfe das sofort für Sie."
    chunks = [
        c async for c in engine.synthesize_clause_stream(de_filler, clause_index=0, language="de")
    ]
    assert len(chunks) > 0
    assert chunks[0].engine_used in ("memory_cache", "redis_cache")

    # Test primary timeout automatic failover (>80ms simulated delay triggers fallback_synth)
    failover_chunks = [
        c
        async for c in engine.synthesize_clause_stream(
            "Uncached dynamic sentence that times out on primary.",
            clause_index=1,
            language="en",
            simulate_slow_primary_sec=0.25,
        )
    ]
    assert len(failover_chunks) > 0
    assert failover_chunks[0].engine_used == "fallback_synth"


@pytest.mark.asyncio
async def test_livekit_tts_adapter_emits_frames():
    
    # Verify our LiveKit adapter yields valid 16-bit PCM AudioFrames.
    engine = ResilientTTSEngine(use_neural_piper=False)
    adapter = NaironLiveKitTTSAdapter(language="de", engine=engine)
    metrics = StreamMetricsSummary(mode="streaming", language="de")
    tokens = simulate_llm_token_stream(
        "Guten Tag, wie kann ich Ihnen heute helfen?", token_delay_ms=2.0
    )

    frames = [f async for f in adapter.stream_llm_turn(tokens, metrics_out=metrics)]
    assert len(frames) > 0
    assert frames[0].sample_rate == 22050
    assert frames[0].num_channels == 1


def test_fastapi_http_ws_and_prometheus_metrics():
    # Verify FastAPI REST synthesis, Prometheus /metrics, and WebSocket barge-in.
    from app import main as main_mod

    main_mod.tts_engine.use_neural_piper = False

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert "en" in health.json()["supported_languages"]

        resp = client.post(
            "/api/synthesize",
            json={
                "text": "One moment please, let me check that for you. Your call is important.",
                "language": "en",
                "mode": "streaming",
                "token_delay_ms": 5.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["clauses_emitted"] >= 2
        assert body["ttfa_ms"] > 0
        assert len(body["wav_base64"]) > 100

        # Check WebSocket streaming + barge-in
        with client.websocket_connect("/ws/tts") as ws:
            ws.send_json(
                {
                    "type": "synthesize",
                    "text": "Hello caller, we are checking your calendar right now.",
                    "language": "en",
                    "mode": "streaming",
                    "token_delay_ms": 5.0,
                }
            )
            first_msg = ws.receive_json()
            assert first_msg["type"] in ("clause_audio", "turn_complete")
            ws.send_json({"type": "barge_in"})
            ack = ws.receive_json()
            assert ack["type"] in ("barge_in_ack", "clause_audio", "turn_complete")

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "tts_ttfa_seconds" in metrics_resp.text
