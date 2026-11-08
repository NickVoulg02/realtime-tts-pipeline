from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from app.metrics import (
    TTS_BARGE_IN_LATENCY_MS,
    TTS_BARGE_IN_TOTAL,
    TTS_TTFA_SECONDS,
)
from app.tts_engine import AudioChunk, ResilientTTSEngine

# Sentence/clause boundary regex: matches ., !, ?, ;, :, or , followed by whitespace or end-of-string
# Avoids splitting inside decimals like "3.5" or common titles like "Dr." / "Mr." / "St."
ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "nr.", "ca.", "z.b.", "bzw."}


@dataclass
class StreamMetricsSummary:
    # Telemetry captured during a single voice turn (Batch or Streaming).

    mode: str
    language: str
    ttfa_ms: float = 0.0
    total_turn_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    total_audio_duration_ms: float = 0.0
    clauses_emitted: int = 0
    chunks_emitted: int = 0
    engines_used: List[str] = field(default_factory=list)
    barge_in_triggered: bool = False
    barge_in_cancel_latency_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.total_audio_duration_ms <= 0:
            return 0.0
        return self.total_compute_time_ms / self.total_audio_duration_ms


async def simulate_llm_token_stream(
    full_text: str,
    token_delay_ms: float = 35.0,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[str]:
    # Simulate an LLM streaming tokens in real-time (~28 tokens/sec at 35ms/token).
    # Splits text into word/punctuation tokens and yields them asynchronously.
    
    tokens = re.findall(r"\S+\s*", full_text)
    for token in tokens:
        if cancel_event and cancel_event.is_set():
            break
        if token_delay_ms > 0:
            await asyncio.sleep(token_delay_ms / 1000.0)
        yield token


class AsyncClauseChunker:
    
    # Buffers incoming LLM tokens and yields natural spoken clauses as soon as a punctuation
    # boundary is hit, minimizing Time-to-First-Audio (TTFA) without sounding choppy.
    

    def __init__(
        self,
        min_words_comma: int = 4,
        min_words_sentence: int = 2,
        max_words_unpunctuated: int = 14,
    ) -> None:
        self.min_words_comma = min_words_comma
        self.min_words_sentence = min_words_sentence
        self.max_words_unpunctuated = max_words_unpunctuated

    def _should_flush(self, buffer: str) -> Optional[tuple[str, str]]:
        # Check if `buffer` contains a speakable clause boundary.
        # Returns (clause_to_emit, remaining_buffer) or None
        
        stripped = buffer.strip()
        if not stripped:
            return None

        words = stripped.split()
        num_words = len(words)

        # Look for strong sentence endings (. ! ? ; :) or soft clause breaks (,)
        for match in re.finditer(r"([.!?;:,])(?=\s|$)", buffer):
            punct = match.group(1)
            end_idx = match.end()
            candidate = buffer[:end_idx].strip()
            candidate_words = candidate.split()

            if not candidate_words:
                continue

            last_word_lower = candidate_words[-1].lower()
            if last_word_lower in ABBREVIATIONS:
                continue

            if punct in ".!?;:" and len(candidate_words) >= self.min_words_sentence:
                return candidate, buffer[end_idx:].lstrip()

            if punct == "," and len(candidate_words) >= self.min_words_comma:
                return candidate, buffer[end_idx:].lstrip()

        # Safety flush if the LLM generates a very long run-on clause without punctuation
        if num_words >= self.max_words_unpunctuated and buffer.endswith(" "):
            return stripped, ""

        return None

    async def chunk_tokens(
        self,
        token_stream: AsyncIterator[str],
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        # Consume streaming tokens and yield ready-to-speak clauses.
        buffer = ""
        async for token in token_stream:
            if cancel_event and cancel_event.is_set():
                return
            buffer += token
            while True:
                split_result = self._should_flush(buffer)
                if split_result is None:
                    break
                clause, buffer = split_result
                if clause:
                    yield clause

        tail = buffer.strip()
        if tail and not (cancel_event and cancel_event.is_set()):
            yield tail


class StreamingTTSPipeline:
    # Coordinates LLM token chunking, bounded-queue backpressure, multi-tier TTS synthesis,
    # and sub-millisecond barge-in cancellation.
    

    def __init__(
        self,
        engine: ResilientTTSEngine,
        max_queue_size: int = 4,
        chunker: Optional[AsyncClauseChunker] = None,
    ) -> None:
        self.engine = engine
        self.max_queue_size = max_queue_size
        self.chunker = chunker or AsyncClauseChunker()

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> int:
        #Immediately discard all pending clauses in the queue upon caller barge-in.
        drained = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    async def stream_from_tokens(
        self,
        token_stream: AsyncIterator[str],
        language: str = "en",
        cancel_event: Optional[asyncio.Event] = None,
        metrics_out: Optional[StreamMetricsSummary] = None,
        simulate_slow_primary_sec: float = 0.0,
        simulate_primary_error: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        # Streaming Mode (Optimized):
        # Concurrently chunks streaming LLM tokens into a bounded asyncio.Queue (backpressure)
        # and synthesizes audio clause-by-clause as tokens arrive.
        
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=self.max_queue_size)
        t_start = time.perf_counter()
        first_audio_emitted = False

        async def _producer() -> None:
            try:
                async for clause in self.chunker.chunk_tokens(token_stream, cancel_event=cancel_event):
                    if cancel_event and cancel_event.is_set():
                        break
                    # Backpressure: awaits if downstream TTS consumer is busy and queue is full
                    await queue.put(clause)
            finally:
                if not (cancel_event and cancel_event.is_set()):
                    await queue.put(None)

        producer_task = asyncio.create_task(_producer())
        clause_idx = 0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    t_cancel = time.perf_counter()
                    self._drain_queue(queue)
                    producer_task.cancel()
                    cancel_ms = (time.perf_counter() - t_cancel) * 1000.0
                    TTS_BARGE_IN_TOTAL.inc()
                    TTS_BARGE_IN_LATENCY_MS.observe(cancel_ms)
                    if metrics_out is not None:
                        metrics_out.barge_in_triggered = True
                        metrics_out.barge_in_cancel_latency_ms = cancel_ms
                    break

                # Poll queue with short timeout so cancel_event is detected even while waiting on LLM tokens
                try:
                    clause = await asyncio.wait_for(queue.get(), timeout=0.02)
                except asyncio.TimeoutError:
                    if producer_task.done() and queue.empty():
                        break
                    continue

                if clause is None:
                    queue.task_done()
                    break

                if metrics_out is not None:
                    metrics_out.clauses_emitted += 1

                async for chunk in self.engine.synthesize_clause_stream(
                    clause_text=clause,
                    clause_index=clause_idx,
                    language=language,
                    cancel_event=cancel_event,
                    simulate_slow_primary_sec=simulate_slow_primary_sec,
                    simulate_primary_error=simulate_primary_error,
                ):
                    if not first_audio_emitted:
                        first_audio_emitted = True
                        ttfa_sec = time.perf_counter() - t_start
                        mode_label = (
                            "cached"
                            if chunk.engine_used in ("redis_cache", "memory_cache")
                            else "streaming"
                        )
                        TTS_TTFA_SECONDS.labels(mode=mode_label, language=language.lower()[:2]).observe(
                            ttfa_sec
                        )
                        if metrics_out is not None:
                            metrics_out.ttfa_ms = ttfa_sec * 1000.0

                    if metrics_out is not None:
                        metrics_out.chunks_emitted += 1
                        metrics_out.total_compute_time_ms += chunk.compute_time_ms
                        metrics_out.total_audio_duration_ms += chunk.audio_duration_ms
                        if chunk.engine_used not in metrics_out.engines_used:
                            metrics_out.engines_used.append(chunk.engine_used)

                    yield chunk

                queue.task_done()
                clause_idx += 1

                if cancel_event and cancel_event.is_set():
                    t_cancel = time.perf_counter()
                    self._drain_queue(queue)
                    producer_task.cancel()
                    cancel_ms = (time.perf_counter() - t_cancel) * 1000.0
                    TTS_BARGE_IN_TOTAL.inc()
                    TTS_BARGE_IN_LATENCY_MS.observe(cancel_ms)
                    if metrics_out is not None:
                        metrics_out.barge_in_triggered = True
                        metrics_out.barge_in_cancel_latency_ms = cancel_ms
                    break

        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            if metrics_out is not None:
                metrics_out.total_turn_time_ms = (time.perf_counter() - t_start) * 1000.0

    async def synthesize_batch_from_tokens(
        self,
        token_stream: AsyncIterator[str],
        language: str = "en",
        cancel_event: Optional[asyncio.Event] = None,
        metrics_out: Optional[StreamMetricsSummary] = None,
    ) -> AsyncIterator[AudioChunk]:
        # Batch Mode (Naive Baseline for Before/After Comparison):
        # Waits for the entire LLM token stream to finish before starting TTS synthesis.
        
        t_start = time.perf_counter()
        collected_tokens: list[str] = []
        async for token in token_stream:
            if cancel_event and cancel_event.is_set():
                return
            collected_tokens.append(token)

        full_text = "".join(collected_tokens).strip()
        if not full_text:
            return

        first_audio_emitted = False
        if metrics_out is not None:
            metrics_out.clauses_emitted = 1

        async for chunk in self.engine.synthesize_clause_stream(
            clause_text=full_text,
            clause_index=0,
            language=language,
            cancel_event=cancel_event,
        ):
            if not first_audio_emitted:
                first_audio_emitted = True
                ttfa_sec = time.perf_counter() - t_start
                TTS_TTFA_SECONDS.labels(mode="batch", language=language.lower()[:2]).observe(ttfa_sec)
                if metrics_out is not None:
                    metrics_out.ttfa_ms = ttfa_sec * 1000.0

            if metrics_out is not None:
                metrics_out.chunks_emitted += 1
                metrics_out.total_compute_time_ms += chunk.compute_time_ms
                metrics_out.total_audio_duration_ms += chunk.audio_duration_ms
                if chunk.engine_used not in metrics_out.engines_used:
                    metrics_out.engines_used.append(chunk.engine_used)

            yield chunk

        if metrics_out is not None:
            metrics_out.total_turn_time_ms = (time.perf_counter() - t_start) * 1000.0

