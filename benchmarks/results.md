# VoiceStream-Eval: Before vs. After Benchmark Report

## Executive Summary

| Metric | Baseline (Batch TTS) | Optimized (Streaming Clause TTS) | Improvement |
| :--- | :--- | :--- | :--- |
| **Mean Time-to-First-Audio (TTFA)** | `1615.0 ms` | **`357.1 ms`** | **-77.9% Latency** |
| **TTFA (Un-cached Dynamic Clauses)** | `1615.0 ms` | **`550.6 ms`** | Concurrent LLM + TTS |
| **TTFA (With Filler Cache Hit)** | `1615.0 ms` | **`260.3 ms`** | Instant Filler Playback |
| **Mean Real-Time Factor (RTF on CPU)** | — | **`0.0816`** (`~12.3x` faster than real-time) | CPU ONNX (`piper-tts`) |
| **Round-Trip Intelligibility (Whisper WER)** | — | **`9.34%` WER** | Verified EN & DE Audio |
| **Barge-In Queue Drain & Cancel Latency** | — | **`0.014 ms`** | Sub-millisecond Interruption |

## Per-Turn Results (English & German)

| Case ID | Lang | Batch TTFA (ms) | Streaming TTFA (ms) | Reduction | RTF | Clauses | Engines Used | Round-Trip WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `en_appointment` | `EN` | 1460.9 ms | **284.2 ms** | **-80.5%** | 0.064 | 2 | `memory_cache, piper_onnx` | 10.5% |
| `en_support_ticket` | `EN` | 1710.4 ms | **418.2 ms** | **-75.6%** | 0.0992 | 3 | `piper_onnx` | 5.0% |
| `en_reschedule` | `EN` | 1602.2 ms | **273.4 ms** | **-82.9%** | 0.0779 | 3 | `memory_cache, piper_onnx` | 0.0% |
| `de_termin_filler` | `DE` | 1574.2 ms | **298.7 ms** | **-81.0%** | 0.061 | 2 | `memory_cache, piper_onnx` | 25.0% |
| `de_rueckruf` | `DE` | 1741.7 ms | **683.1 ms** | **-60.8%** | 0.1442 | 3 | `piper_onnx` | 5.0% |
| `de_kalender` | `DE` | 1600.6 ms | **185.1 ms** | **-88.4%** | 0.0434 | 3 | `memory_cache, piper_onnx` | 10.5% |
