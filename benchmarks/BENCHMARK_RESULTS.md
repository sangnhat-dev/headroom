# Headroom SDK Benchmark Report

Generated: 2026-03-08T02:30:24.698238

## Environment

- **Machine**: x86_64
- **Processor**: x86_64
- **Python**: 3.12.3

## Results Summary

| Test | Mean | StdDev | Min | Max | Target | Status |
|------|------|--------|-----|-----|--------|--------|
| `test_compress_100_items` | 283.16ms | 1.13ms | 281.31ms | 284.28ms | 500.00ms | PASS |
| `test_compress_1000_items` | 2.88s | 18.06ms | 2.86s | 2.90s | 5.00s | PASS |
| `test_compress_10000_items` | 39.08s | 200.83ms | 38.87s | 39.37s | 60.00s | PASS |
| `test_analyze_log_entries` | 2.01s | 25.90ms | 1.98s | 2.05s | - | - |
| `test_analyze_metrics_with_anomalies` | 1.39s | 7.89ms | 1.38s | 1.40s | - | - |
| `test_multiple_tool_outputs` | 485.90ms | 1.97ms | 483.41ms | 487.87ms | - | - |
| `test_date_extraction` | 173.6us | 6.3us | 166.6us | 260.1us | 1.00ms | PASS |
| `test_hash_computation` | 584.3us | 14.2us | 543.1us | 770.8us | 1.00ms | PASS |
| `test_whitespace_normalization` | 111.4us | 4.9us | 106.6us | 185.6us | - | - |
| `test_long_system_prompt` | 628.8us | 13.2us | 588.5us | 749.3us | - | - |
| `test_multiple_system_messages` | 96.3us | 5.9us | 91.7us | 160.4us | - | - |
| `test_window_50_turns` | 8.51ms | 118.2us | 8.23ms | 8.74ms | 15.00ms | PASS |
| `test_window_200_turns` | 16.18ms | 239.3us | 15.68ms | 16.66ms | 30.00ms | PASS |
| `test_window_no_drop_needed` | 1.14ms | 14.2us | 1.12ms | 1.32ms | - | - |
| `test_window_aggressive_drop` | 8.47ms | 165.9us | 8.29ms | 8.95ms | - | - |
| `test_window_rag_context` | 359.0us | 11.1us | 350.2us | 641.7us | - | - |
| `test_pipeline_simple` | 363.4us | 20.2us | 347.5us | 842.9us | 5.00ms | PASS |
| `test_pipeline_agentic` | 11.89s | 57.09ms | 11.84s | 11.99s | 30.00s | PASS |
| `test_pipeline_rag` | 1.88ms | 31.5us | 1.81ms | 2.16ms | 50.00ms | PASS |
| `test_compress_log_100_items` | 205.9us | 15.5us | 198.1us | 422.3us | 2.00ms | PASS |
| `test_compress_log_1000_items` | 1.58ms | 52.7us | 1.53ms | 2.65ms | 15.00ms | PASS |
| `test_compress_diagnostics_200` | 418.2us | 12.3us | 404.1us | 722.4us | 5.00ms | PASS |
| `test_compress_stack_traces_100` | 158.6us | 6.2us | 153.0us | 275.3us | 3.00ms | PASS |
| `test_compress_multi_turn_5_turns` | 744.5us | 16.9us | 717.9us | 970.2us | 10.00ms | PASS |
| `test_compress_multi_turn_20_turns` | 3.00ms | 167.1us | 2.92ms | 4.58ms | 40.00ms | PASS |
| `test_deeply_nested_schema_5_levels` | 115.5us | 4.6us | 111.4us | 195.0us | 5.00ms | PASS |
| `test_compress_log_with_errors_100` | 112.29ms | 832.9us | 111.43ms | 113.78ms | 150.00ms | PASS |
| `test_compress_log_with_errors_500` | 596.46ms | 12.55ms | 589.66ms | 618.77ms | 750.00ms | PASS |
| `test_compress_diagnostics_100` | 129.85ms | 473.2us | 129.09ms | 130.41ms | 150.00ms | PASS |
| `test_compress_metrics_with_spike` | 103.85ms | 806.4us | 102.80ms | 105.20ms | 150.00ms | PASS |
| `test_plain_text_passthrough_1000_chars` | 18.4us | 1.6us | 17.7us | 69.0us | 5.00ms | PASS |

## Summary

- **Passed**: 22/22 (100%)
- **Failed**: 0/22 (0%)

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| SmartCrusher (100 items) | < 500ms | Statistical BM25 analysis |
| SmartCrusher (1000 items) | < 5s | Statistical BM25 analysis |
| SmartCrusher (10000 items) | < 60s | Stress test |
| CacheAligner | < 1ms | Date extraction + hash |
| RollingWindow (50 turns) | < 15ms | Long conversation |
| RollingWindow (200 turns) | < 30ms | Stress test |
| BM25Scorer (batch 100) | < 1ms | Zero dependencies |
| HybridScorer (batch 100) | < 50ms | With embeddings |
| ToolCrusher log 100 items | < 2ms | Fixed-rule compression |
| ToolCrusher log 1000 items | < 15ms | Fixed-rule compression |
| ToolCrusher 5-turn agent | < 10ms | Multi-step conversation |
| ToolCrusher 20-turn agent | < 40ms | Long agentic session |
| SmartCrusher log w/errors 100 | < 150ms | Error-preserving statistical |
| SmartCrusher plain text 1000c | < 5ms | Fast passthrough path |
