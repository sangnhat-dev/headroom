# Headroom Quality & Token-Reduction Report

Generated: 2026-03-08T02:47:12+00:00  
Platform: x86_64 · Python 3.12.3

> **How to read this table**
> - **Tokens (No HR)**: tokens sent to the LLM *without* Headroom
> - **Tokens (With HR)**: tokens sent to the LLM *with* Headroom
> - **Reduction**: percentage fewer tokens when Headroom is enabled
> - **Info Recall**: fraction of key facts still present after compression
> - **Critical Items**: errors / anomalies that *must not* be lost (retained/total)
> - **Pass**: quality threshold met (recall ≥ 80 %, critical retention = 100 %)

## Results by Scenario

| Scenario | Tokens (No HR) | Tokens (With HR) | Reduction | Info Recall | Critical Items | Pass |
|----------|---------------:|------------------:|----------:|:-----------:|:--------------:|:----:|
| `Search Results (50 items)` | 4,094 | 844 | -79.4% | 83% | 3/3 | ✅ PASS |
| `Search Results (200 items)` | 16,199 | 1,681 | -89.6% | 100% | 7/7 | ✅ PASS |
| `Search Results (500 items)` | 40,277 | 2,494 | -93.8% | 94% | 13/13 | ✅ PASS |
| `Search Results (2K items)` | 161,308 | 5,027 | -96.9% | 98% | 43/43 | ✅ PASS |
| `Structured Logs (100 entries)` | 6,245 | 1,743 | -72.1% | 100% | 7/7 | ✅ PASS |
| `Structured Logs (500 entries)` | 30,805 | 6,353 | -79.4% | 92% | 35/35 | ✅ PASS |
| `Structured Logs (2K entries)` | 122,894 | 23,536 | -80.8% | 98% | 140/140 | ✅ PASS |
| `API Responses (100 items)` | 6,350 | 1,204 | -81.0% | 100% | n/a | ✅ PASS |
| `API Responses (500 items)` | 31,047 | 1,171 | -96.2% | 100% | n/a | ✅ PASS |
| `Database Rows (500 rows (metrics))` | 17,469 | 762 | -95.6% | 100% | 3/3 | ✅ PASS |
| `Database Rows (1K rows (mixed))` | 38,719 | 4,480 | -88.4% | 100% | 1/1 | ✅ PASS |
| `Agentic Conversation (5 turns / 30 items each)` | 14,173 | 8,892 | -37.3% | 100% | n/a | ✅ PASS |
| `Agentic Conversation (15 turns / 50 items each)` | 67,341 | 39,516 | -41.3% | 100% | n/a | ✅ PASS |
| `RAG Conversation (3K context)` | 3,461 | 3,461 | - | 100% | 2/2 | ✅ PASS |
| `RAG Conversation (10K context)` | 10,562 | 10,562 | - | 100% | 2/2 | ✅ PASS |

## Summary

| Metric | Value |
|--------|-------|
| Scenarios run | 15 |
| Quality passed | 15/15 (100%) |
| Tokens **without** Headroom | 570,944 |
| Tokens **with** Headroom | 111,726 |
| Total tokens saved | 459,218 (80.4% reduction) |
| Avg information recall | 97.7% |
| Critical items retained | 256/256 (100%) |

## Quality Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| Information recall | ≥ 80% | Key facts still present after compression |
| Critical retention | = 100% | Errors / anomalies must never be lost |
| Meaningful compression | ≥ 10% | Minimum token reduction to count as compressed |

## Interpretation

Headroom reduces token usage by compressing large tool outputs (JSON arrays,
log files, API responses) while preserving the information that matters most.

The **Info Recall** metric verifies that probe facts — specific error codes,
anomalous values, UUIDs, and critical log messages — are still present in the
compressed output and therefore visible to the LLM.

Critical items (CRITICAL / ERROR log entries, statistical anomalies) are **never**
dropped by SmartCrusher's error-preservation and change-point detection algorithms.

## Compressor Used

All scenarios use `SmartCrusher` (BM25 relevance tier) — no sentence-transformers
model download or API key required.  The BM25 tier runs fully offline.

