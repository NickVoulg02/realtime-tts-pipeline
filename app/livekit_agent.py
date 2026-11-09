from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from app.pipeline import StreamingTTSPipeline, StreamMetricsSummary, simulate_llm_token_stream
from app.tts_engine import AudioChunk, ResilientTTSEngine

logger = logging.getLogger(__name__)


@dataclass
class LiveKitAudioFrame:
    # Lightweight representation compatible with `livekit.rtc.AudioFrame`.

    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int


class NaironLiveKitTTSAdapter:
    # Adapts our `StreamingTTSPipeline` and `ResilientTTSEngine` into a LiveKit-compatible
    # frame emitter (`livekit.rtc.AudioFrame` when `livekit-rtc` is loaded).

    def __init__(
        self,
        language: str = "en",
        sample_rate: int = 22050,
        num_channels: int = 1,
        engine: Optional[ResilientTTSEngine] = None,
    ) -> None:
        self.language = language
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.engine = engine or ResilientTTSEngine(sample_rate=sample_rate)
        self.pipeline = StreamingTTSPipeline(engine=self.engine, max_queue_size=4)

    @staticmethod
    def _to_rtc_audio_frame(chunk: AudioChunk):
        # Convert our `AudioChunk` (16-bit mono PCM) into `livekit.rtc.AudioFrame` if installed.
        samples_per_channel = len(chunk.pcm_bytes) // 2
        try:
            from livekit import rtc

            return rtc.AudioFrame(
                data=chunk.pcm_bytes,
                sample_rate=chunk.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_channel,
            )
        except Exception:
            return LiveKitAudioFrame(
                data=chunk.pcm_bytes,
                sample_rate=chunk.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_channel,
            )

    async def stream_llm_turn(
        self,
        token_stream: AsyncIterator[str],
        cancel_event: Optional[asyncio.Event] = None,
        metrics_out: Optional[StreamMetricsSummary] = None,
    ) -> AsyncIterator[object]:
        
        # Consume an async LLM token stream and yield LiveKit `rtc.AudioFrame` objects
        # with dynamic clause chunking, Redis filler caching, and barge-in support.
        
        async for chunk in self.pipeline.stream_from_tokens(
            token_stream=token_stream,
            language=self.language,
            cancel_event=cancel_event,
            metrics_out=metrics_out,
        ):
            yield self._to_rtc_audio_frame(chunk)


async def run_local_demo() -> None:
    # Run a self-contained English & German LiveKit frame streaming simulation.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = ResilientTTSEngine()
    await engine.warm_up_fillers(languages=("en", "de"))

    for lang, prompt in [
        (
            "en",
            "One moment please, let me check that for you. Your appointment is confirmed for tomorrow at ten AM.",
        ),
        (
            "de",
            "Einen Moment bitte, ich prüfe das sofort für Sie. Ihr Termin für morgen um zehn Uhr ist bestätigt.",
        ),
    ]:
        adapter = NaironLiveKitTTSAdapter(language=lang, engine=engine)
        metrics = StreamMetricsSummary(mode="streaming", language=lang)
        tokens = simulate_llm_token_stream(prompt, token_delay_ms=30.0)

        frame_count = 0
        async for frame in adapter.stream_llm_turn(tokens, metrics_out=metrics):
            frame_count += 1

        logger.info(
            "[LiveKit Adapter Demo | lang=%s] Emitted %d AudioFrames across %d clauses | TTFA=%.1fms | RTF=%.3f | Engines=%s",
            lang,
            frame_count,
            metrics.clauses_emitted,
            metrics.ttfa_ms,
            metrics.rtf,
            metrics.engines_used,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveKit TTS Adapter Worker & Demo")
    parser.add_argument("--demo", action="store_true", help="Run standalone LiveKit AudioFrame demo")
    args = parser.parse_args()
    asyncio.run(run_local_demo())

