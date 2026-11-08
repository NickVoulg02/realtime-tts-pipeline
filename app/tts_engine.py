from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

import httpx
import numpy as np

from app.metrics import (
    TTS_CACHE_HITS_TOTAL,
    TTS_CLAUSES_SYNTHESIZED_TOTAL,
    TTS_FALLBACK_TOTAL,
    TTS_RTF_RATIO,
)

logger = logging.getLogger(__name__)

# Official HuggingFace URLs for lightweight CPU Piper ONNX models (English & German)
PIPER_VOICE_URLS: Dict[str, Dict[str, str]] = {
    "en": {
        "name": "en_US-lessac-medium",
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "de": {
        "name": "de_DE-thorsten-medium",
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
    },
}

# Pre-warmed conversational fillers used by voice agents to mask CRM/calendar tool-call latency
DEFAULT_FILLER_PHRASES: Dict[str, list[str]] = {
    "en": [
        "One moment please, let me check that for you.",
        "Let me look up your appointment in the calendar.",
        "Got it, just a second while I pull up your account.",
    ],
    "de": [
        "Einen Moment bitte, ich prüfe das sofort für Sie.",
        "Ich schaue kurz in Ihren Terminkalender.",
        "Verstanden, einen Augenblick bitte.",
    ],
}


@dataclass
class AudioChunk:
    # ""Represents a streamed frame of 16-bit mono PCM audio with synthesis telemetry.

    pcm_bytes: bytes
    sample_rate: int
    clause_text: str
    clause_index: int
    engine_used: str  # "redis_cache" | "memory_cache" | "piper_onnx" | "fallback_synth"
    compute_time_ms: float
    audio_duration_ms: float
    is_final_chunk: bool = False


def normalize_phrase_key(text: str, language: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    digest = hashlib.sha256(f"{language}:{cleaned}".encode("utf-8")).hexdigest()[:24]
    return f"tts:filler:{language}:{digest}"


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class ResilientTTSEngine:
    # Multi-tier TTS engine designed for real-time phone voice agents:
    # Tier 0: Redis / In-Memory Filler Cache (<5ms instant response for common phrases)
    # Tier 1: Local CPU Neural TTS via Piper ONNX (English & German)
    # Tier 2: Automatic timeout/error failover to fast deterministic PCM synthesizer

    def __init__(
        self,
        models_dir: str | Path = "models",
        redis_url: Optional[str] = None,
        timeout_sec: float = 2.5,
        sample_rate: int = 22050,
        frame_duration_ms: int = 100,
        use_neural_piper: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.timeout_sec = timeout_sec
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.use_neural_piper = use_neural_piper

        self._piper_voices: Dict[str, object] = {}
        self._memory_cache: Dict[str, bytes] = {}
        self._redis_client = None
        self._redis_checked = False

    async def _get_redis(self):
        # Lazily initialize async Redis connection; degrades gracefully to memory cache if offline."
        if self._redis_checked:
            return self._redis_client
        self._redis_checked = True
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                self.redis_url,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
            )
            await client.ping()
            self._redis_client = client
            logger.info("Connected to Redis filler cache at %s", self.redis_url)
        except Exception:
            self._redis_client = None
            logger.info("Redis unavailable; using in-memory filler cache fallback.")
        return self._redis_client

    async def get_cached_pcm(self, text: str, language: str) -> tuple[Optional[bytes], str]:
        # Look up pre-synthesized filler audio in Redis first, then local memory.
        key = normalize_phrase_key(text, language)
        redis_client = await self._get_redis()
        if redis_client is not None:
            try:
                cached = await redis_client.get(key)
                if cached:
                    return cached, "redis_cache"
            except Exception:
                pass
        if key in self._memory_cache:
            return self._memory_cache[key], "memory_cache"
        return None, ""

    async def set_cached_pcm(self, text: str, language: str, pcm_bytes: bytes) -> None:
        # Store synthesized filler audio in both memory and Redis (with 24h TTL).
        key = normalize_phrase_key(text, language)
        self._memory_cache[key] = pcm_bytes
        redis_client = await self._get_redis()
        if redis_client is not None:
            try:
                await redis_client.set(key, pcm_bytes, ex=86400)
            except Exception:
                pass

    async def ensure_voice_loaded(self, language: str) -> Optional[object]:
        # Download (if needed) and load the Piper ONNX voice model for the target language.
        lang = language.lower()[:2]
        if lang not in PIPER_VOICE_URLS:
            lang = "en"
        if not self.use_neural_piper:
            return None
        if lang in self._piper_voices:
            return self._piper_voices[lang]

        voice_info = PIPER_VOICE_URLS[lang]
        onnx_path = self.models_dir / f"{voice_info['name']}.onnx"
        json_path = self.models_dir / f"{voice_info['name']}.onnx.json"

        if not onnx_path.exists() or not json_path.exists():
            logger.info("Downloading Piper CPU voice '%s' for language '%s'...", voice_info["name"], lang)
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                for target_path, url in [(json_path, voice_info["json"]), (onnx_path, voice_info["onnx"])]:
                    if not target_path.exists():
                        resp = await client.get(url)
                        resp.raise_for_status()
                        target_path.write_bytes(resp.content)

        def _load_sync():
            from piper.voice import PiperVoice

            return PiperVoice.load(str(onnx_path), config_path=str(json_path))

        try:
            voice = await asyncio.to_thread(_load_sync)
            self._piper_voices[lang] = voice
            return voice
        except Exception as exc:
            logger.warning("Failed to load Piper voice (%s), using fallback synth: %s", lang, exc)
            return None

    async def warm_up_fillers(self, languages: tuple[str, ...] = ("en", "de")) -> int:
        # Pre-synthesize and cache common conversational fillers for instant <5ms playback.
        count = 0
        for lang in languages:
            await self.ensure_voice_loaded(lang)
            for phrase in DEFAULT_FILLER_PHRASES.get(lang, []):
                existing, _ = await self.get_cached_pcm(phrase, lang)
                if existing is None:
                    pcm, _, _ = await self._synthesize_clause_bytes(phrase, lang)
                    await self.set_cached_pcm(phrase, lang, pcm)
                count += 1
        return count

    def _synthesize_piper_sync(self, voice: object, text: str) -> tuple[bytes, int]:
        # Run Piper ONNX synthesis synchronously inside a worker thread across piper-tts versions.
        sample_rate = getattr(getattr(voice, "config", None), "sample_rate", self.sample_rate)

        # Compatibility across piper-tts 1.2 (synthesize_stream_raw) and 1.3-1.8 (synthesize / synthesize_wav)
        if hasattr(voice, "synthesize_stream_raw"):
            chunks = list(voice.synthesize_stream_raw(text))
            return b"".join(chunks), sample_rate

        if hasattr(voice, "synthesize"):
            raw_parts: list[bytes] = []
            for item in voice.synthesize(text):
                if isinstance(item, (bytes, bytearray)):
                    raw_parts.append(bytes(item))
                elif hasattr(item, "audio_int16_bytes"):
                    raw_parts.append(item.audio_int16_bytes)
                elif hasattr(item, "audio_bytes"):
                    raw_parts.append(item.audio_bytes)
            if raw_parts:
                return b"".join(raw_parts), sample_rate

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as rf:
            return rf.readframes(rf.getnframes()), rf.getframerate()

    def _synthesize_fallback_sync(self, text: str) -> tuple[bytes, int]:
        # Generates natural syllable-cadence harmonic PCM audio when primary TTS times out or is offline.
        words = max(1, len(text.split()))
        duration_sec = min(4.0, max(0.25, words * 0.18))
        num_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)

        # Fundamental frequency (150Hz) + first harmonic (300Hz) modulated at syllable rate (4.5Hz)
        carrier = 0.6 * np.sin(2 * np.pi * 150.0 * t) + 0.3 * np.sin(2 * np.pi * 300.0 * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t)
        pcm_int16 = (carrier * envelope * 12000).astype(np.int16)
        return pcm_int16.tobytes(), self.sample_rate

    async def _synthesize_clause_bytes(
        self,
        text: str,
        language: str,
        simulate_slow_primary_sec: float = 0.0,
        simulate_primary_error: bool = False,
    ) -> tuple[bytes, int, str]:
        # Synthesize a single clause with automatic timeout & error failover.
        lang = language.lower()[:2]

        # Check Filler Cache (<5ms instant hit)
        cached_pcm, cache_source = await self.get_cached_pcm(text, lang)
        if cached_pcm is not None:
            TTS_CACHE_HITS_TOTAL.labels(language=lang).inc()
            return cached_pcm, self.sample_rate, cache_source

        # Primary Engine (Piper ONNX CPU) wrapped in asyncio.wait_for timeout guard
        try:
            async def _run_primary() -> tuple[bytes, int, str]:
                if simulate_primary_error:
                    raise RuntimeError("Simulated primary TTS engine failure")
                if simulate_slow_primary_sec > 0:
                    await asyncio.sleep(simulate_slow_primary_sec)

                voice = await self.ensure_voice_loaded(lang)
                if voice is None:
                    pcm, sr = await asyncio.to_thread(self._synthesize_fallback_sync, text)
                    return pcm, sr, "fallback_synth"

                pcm, sr = await asyncio.to_thread(self._synthesize_piper_sync, voice, text)
                return pcm, sr, "piper_onnx"

            pcm_bytes, sr, engine_used = await asyncio.wait_for(_run_primary(), timeout=self.timeout_sec)
            TTS_CLAUSES_SYNTHESIZED_TOTAL.labels(engine=engine_used, language=lang).inc()
            return pcm_bytes, sr, engine_used

        except asyncio.TimeoutError:
            TTS_FALLBACK_TOTAL.labels(language=lang, reason="timeout").inc()
            logger.warning("Primary TTS timed out (>%.2fs); failing over to fast fallback.", self.timeout_sec)
        except Exception as exc:
            TTS_FALLBACK_TOTAL.labels(language=lang, reason="error").inc()
            logger.warning("Primary TTS error (%s); failing over to fast fallback.", exc)

        # 3. Resilient Fallback Synthesizer
        pcm_bytes, sr = await asyncio.to_thread(self._synthesize_fallback_sync, text)
        TTS_CLAUSES_SYNTHESIZED_TOTAL.labels(engine="fallback_synth", language=lang).inc()
        return pcm_bytes, sr, "fallback_synth"

    async def synthesize_clause_stream(
        self,
        clause_text: str,
        clause_index: int = 0,
        language: str = "en",
        cancel_event: Optional[asyncio.Event] = None,
        simulate_slow_primary_sec: float = 0.0,
        simulate_primary_error: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        
        # Synthesize a clause and stream it out in fixed PCM frames (e.g. 100ms chunks),
        # checking `cancel_event` before every frame so caller barge-in halts playback immediately.
        if cancel_event and cancel_event.is_set():
            return

        t0 = time.perf_counter()
        pcm_bytes, sr, engine_used = await self._synthesize_clause_bytes(
            clause_text,
            language=language,
            simulate_slow_primary_sec=simulate_slow_primary_sec,
            simulate_primary_error=simulate_primary_error,
        )
        compute_ms = (time.perf_counter() - t0) * 1000.0

        total_samples = len(pcm_bytes) // 2  # 16-bit (2 bytes per sample)
        audio_duration_ms = (total_samples / float(sr)) * 1000.0 if sr > 0 else 0.0

        if audio_duration_ms > 0:
            rtf = (compute_ms / 1000.0) / (audio_duration_ms / 1000.0)
            TTS_RTF_RATIO.labels(engine=engine_used, language=language.lower()[:2]).observe(rtf)

        # Slice PCM into 100ms transport frames so WebSockets/LiveKit can stream and interrupt cleanly
        bytes_per_frame = max(2, int(sr * (self.frame_duration_ms / 1000.0)) * 2)
        total_len = len(pcm_bytes)

        for offset in range(0, total_len, bytes_per_frame):
            if cancel_event and cancel_event.is_set():
                return
            frame = pcm_bytes[offset : offset + bytes_per_frame]
            is_last = (offset + bytes_per_frame) >= total_len
            frame_samples = len(frame) // 2
            frame_duration_ms = (frame_samples / float(sr)) * 1000.0
            yield AudioChunk(
                pcm_bytes=frame,
                sample_rate=sr,
                clause_text=clause_text,
                clause_index=clause_index,
                engine_used=engine_used,
                compute_time_ms=compute_ms if offset == 0 else 0.0,
                audio_duration_ms=frame_duration_ms,
                is_final_chunk=is_last,
            )
            await asyncio.sleep(0)  

