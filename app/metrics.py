"""Prometheus observability metrics for real-time voice synthesis."""

from prometheus_client import Counter, Histogram

# Time-to-First-Audio (TTFA): Seconds from request/stream start until the first PCM audio chunk is emitted
TTS_TTFA_SECONDS = Histogram(
    "tts_ttfa_seconds",
    "Time to First Audio (TTFA) in seconds across synthesis modes and languages",
    labelnames=["mode", "language"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0),
)

# Real-Time Factor (RTF): compute_time / audio_duration (< 1.0 means faster than real-time)
TTS_RTF_RATIO = Histogram(
    "tts_rtf_ratio",
    "Real-Time Factor (synthesis compute time divided by generated audio duration)",
    labelnames=["engine", "language"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0),
)

# Barge-in (Caller interruption) cancellation latency in milliseconds
TTS_BARGE_IN_LATENCY_MS = Histogram(
    "tts_barge_in_latency_ms",
    "Time taken in milliseconds to cancel active TTS stream and drain audio queue on barge-in",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0),
)

TTS_BARGE_IN_TOTAL = Counter(
    "tts_barge_in_total",
    "Total number of caller barge-in (interruption) events handled",
)

# Cache & Resilience counters
TTS_CACHE_HITS_TOTAL = Counter(
    "tts_cache_hits_total",
    "Total number of instant filler phrase hits served from Redis/memory cache",
    labelnames=["language"],
)

TTS_FALLBACK_TOTAL = Counter(
    "tts_fallback_total",
    "Total number of TTS failovers triggered due to primary engine timeout or error",
    labelnames=["language", "reason"],
)

TTS_CLAUSES_SYNTHESIZED_TOTAL = Counter(
    "tts_clauses_synthesized_total",
    "Total number of spoken clauses synthesized",
    labelnames=["engine", "language"],
)

