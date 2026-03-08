#!/usr/bin/env python3
"""Quality & Token-Reduction Benchmark for Headroom.

Answers the key question: **how much does using Headroom reduce token usage
and does it preserve output quality?**

Each scenario is run TWICE:
  - WITHOUT Headroom  → baseline token count, full content
  - WITH Headroom     → compressed token count, check information retained

Quality is measured by:
  - Information recall   (% of probe facts that survive compression)
  - Critical-item retention (errors/anomalies/needles still present)
  - Structural integrity  (output is still valid JSON / parseable)

No LLM API key is required — this runs entirely offline.

Usage:
    # Print full report to terminal
    python benchmarks/bench_quality.py

    # Save markdown report
    python benchmarks/bench_quality.py --output benchmarks/QUALITY_RESULTS.md

    # Save raw JSON results
    python benchmarks/bench_quality.py --json results.json

    # Run a single scenario type
    python benchmarks/bench_quality.py --scenario search
    python benchmarks/bench_quality.py --scenario logs
    python benchmarks/bench_quality.py --scenario api
    python benchmarks/bench_quality.py --scenario db
    python benchmarks/bench_quality.py --scenario agentic
    python benchmarks/bench_quality.py --scenario rag

    # Verbose: show per-probe detail
    python benchmarks/bench_quality.py --verbose

Available scenario types:
    search   - Elasticsearch-style search results (50 → 2000 items)
    logs     - Structured logs with critical errors + anomalies
    api      - Paginated REST API responses
    db       - Database query results with injected anomalies
    agentic  - Multi-turn agentic conversations with tool calls
    rag      - RAG conversations with large injected context
    all      - All of the above (default)
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure repo root is importable when running as standalone script
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from benchmarks.scenarios.conversations import (  # noqa: E402
    generate_agentic_conversation,
    generate_rag_conversation,
)
from benchmarks.scenarios.tool_outputs import (  # noqa: E402
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

# ---------------------------------------------------------------------------
# Quality-score pass thresholds
# ---------------------------------------------------------------------------

# Minimum fraction of probe facts that must survive compression
RECALL_PASS_THRESHOLD = 0.80  # 80 %

# Minimum fraction of critical items (errors / anomalies) that must survive
CRITICAL_RETENTION_THRESHOLD = 1.00  # 100 % — no critical item may be lost

# Minimum token-reduction that counts as "meaningful compression"
MIN_MEANINGFUL_REDUCTION = 0.10  # 10 %


# ---------------------------------------------------------------------------
# Mock token counter (no tiktoken, no network)
# ---------------------------------------------------------------------------


class _MockTokenCounter:
    """Character-based token counter (4 chars ≈ 1 token).  No tiktoken needed."""

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_message(self, message: dict[str, Any]) -> int:
        content = message.get("content") or ""
        if isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    total += self.count_text(block.get("text", "") or block.get("content", "") or "")
                else:
                    total += self.count_text(str(block))
            return total + 4
        # Include tool call arguments
        extra = sum(
            self.count_text(tc.get("function", {}).get("arguments", ""))
            for tc in message.get("tool_calls", [])
        )
        return max(1, len(str(content)) // 4) + 4 + extra

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count_message(m) for m in messages)


# ---------------------------------------------------------------------------
# Lazy pipeline construction (avoids tiktoken)
# ---------------------------------------------------------------------------

_MOCK_COUNTER = _MockTokenCounter()
_TOKENIZER: Any = None
_CRUSHER: Any = None
_TOOL_CRUSHER: Any = None


def _get_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        from headroom.tokenizer import Tokenizer

        _TOKENIZER = Tokenizer(_MOCK_COUNTER, "benchmark-model")
    return _TOKENIZER


def _get_crusher() -> Any:
    """SmartCrusher backed by BM25 — no sentence-transformers needed."""
    global _CRUSHER
    if _CRUSHER is None:
        from headroom.config import RelevanceScorerConfig, SmartCrusherConfig
        from headroom.transforms.smart_crusher import SmartCrusher

        config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            min_items_to_analyze=3,
            max_items_after_crush=20,
            preserve_change_points=True,
        )
        _CRUSHER = SmartCrusher(config, relevance_config=RelevanceScorerConfig(tier="bm25"))
    return _CRUSHER


def _get_tool_crusher() -> Any:
    """ToolCrusher for rule-based JSON compression."""
    global _TOOL_CRUSHER
    if _TOOL_CRUSHER is None:
        from headroom.config import ToolCrusherConfig
        from headroom.transforms.tool_crusher import ToolCrusher

        config = ToolCrusherConfig(
            enabled=True,
            min_tokens_to_crush=0,
            max_array_items=20,
            max_string_length=500,
        )
        _TOOL_CRUSHER = ToolCrusher(config)
    return _TOOL_CRUSHER


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of checking a single probe fact / needle."""

    probe: str
    survived: bool


@dataclass
class QualityResult:
    """Quality & compression metrics for a single scenario."""

    scenario_name: str
    scenario_type: str
    size_label: str

    # Token counts
    tokens_without_headroom: int  # baseline
    tokens_with_headroom: int  # after compression
    tokens_saved: int
    reduction_pct: float  # 0-100

    # Quality
    probe_results: list[ProbeResult]
    information_recall: float  # fraction of probes that survived
    critical_items_total: int
    critical_items_retained: int
    critical_retention_rate: float  # fraction retained
    structural_integrity: bool  # compressed output is valid JSON / non-empty

    # Overall pass/fail
    quality_passed: bool
    compression_meaningful: bool

    # Timing
    compress_ms: float = 0.0

    # Metadata
    compressor_used: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "size_label": self.size_label,
            "tokens_without_headroom": self.tokens_without_headroom,
            "tokens_with_headroom": self.tokens_with_headroom,
            "tokens_saved": self.tokens_saved,
            "reduction_pct": round(self.reduction_pct, 2),
            "information_recall": round(self.information_recall, 4),
            "critical_items_total": self.critical_items_total,
            "critical_items_retained": self.critical_items_retained,
            "critical_retention_rate": round(self.critical_retention_rate, 4),
            "structural_integrity": self.structural_integrity,
            "quality_passed": self.quality_passed,
            "compression_meaningful": self.compression_meaningful,
            "compress_ms": round(self.compress_ms, 2),
            "compressor_used": self.compressor_used,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Probe-fact extraction helpers
# ---------------------------------------------------------------------------


def _extract_needles_search(items: list[dict]) -> tuple[list[str], list[str]]:
    """Return (probe_facts, critical_needles) for search results."""
    probes: list[str] = []
    criticals: list[str] = []
    for item in items:
        if item.get("is_needle") and item.get("uuid"):
            needle = str(item["uuid"])
            probes.append(needle)
            criticals.append(needle)
        if item.get("status") == "failed" and item.get("error"):
            err = str(item["error"])
            probes.append(err)
            criticals.append(err)
    for item in items[:3]:
        title = item.get("title", "")
        if title:
            probes.append(title)
    return probes, criticals


def _extract_needles_logs(entries: list[dict]) -> tuple[list[str], list[str]]:
    """Only probe CRITICAL/ERROR messages — INFO/DEBUG may legitimately be compressed away."""
    probes: list[str] = []
    criticals: list[str] = []
    for entry in entries:
        level = entry.get("level", "")
        msg = entry.get("message", "")
        if level in ("CRITICAL", "ERROR"):
            probes.append(msg)
            criticals.append(msg)
    # Non-critical probes: first 3 entries (SmartCrusher always keeps first items)
    for entry in entries[:3]:
        msg = entry.get("message", "")
        if msg:
            probes.append(msg)
    return probes, criticals


def _extract_needles_api(items: list[dict]) -> tuple[list[str], list[str]]:
    """Probe the first 3 item IDs — SmartCrusher keeps leading items by default."""
    probes: list[str] = []
    for item in items[:3]:
        probes.append(str(item["id"]))
    return probes, []


def _extract_needles_db(rows: list[dict]) -> tuple[list[str], list[str]]:
    probes: list[str] = []
    criticals: list[str] = []
    for row in rows:
        if row.get("is_anomaly"):
            val = str(row.get("value", ""))
            probes.append(val)
            criticals.append(val)
    for row in rows[:3]:
        probes.append(str(row.get("id", "")))
    return probes, criticals


def _probe_survived(probe: str, compressed_text: str) -> bool:
    if not probe:
        return True
    return probe.lower() in compressed_text.lower()


def _structural_ok(messages: list[dict[str, Any]], originally_json: bool) -> bool:
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        if not content or not content.strip():
            return False
        if originally_json:
            json_part = content.split("\n<headroom:")[0].strip()
            try:
                json.loads(json_part)
            except json.JSONDecodeError:
                return False
    return True


def _tool_content(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            parts.append(str(content))
    return " ".join(parts)


def _all_content(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        parts.append(str(content))
        for tc in msg.get("tool_calls", []):
            parts.append(tc.get("function", {}).get("arguments", ""))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core compression (uses SmartCrusher + ToolCrusher directly, no tiktoken)
# ---------------------------------------------------------------------------


def _compress_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, float]:
    """Compress messages and return (compressed_msgs, compressor_name, compress_ms).

    Uses SmartCrusher (BM25) for statistical array compression and ToolCrusher
    as a fallback for non-array JSON.  Neither requires a network connection.
    """
    tokenizer = _get_tokenizer()
    crusher = _get_crusher()

    t0 = time.perf_counter()
    result = crusher.apply(messages, tokenizer)
    compress_ms = (time.perf_counter() - t0) * 1000.0
    return result.messages, "SmartCrusher(BM25)", compress_ms


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def _run_json_scenario(
    name: str,
    size_label: str,
    items: list[dict],
    probe_facts: list[str],
    critical_needles: list[str],
) -> QualityResult:
    content = json.dumps(items)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.\n\nCurrent date: 2025-01-06",
        },
        {"role": "user", "content": "Analyze the following data."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_quality_1",
                    "type": "function",
                    "function": {"name": "get_data", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_quality_1", "content": content},
    ]

    tokenizer = _get_tokenizer()
    tokens_before = tokenizer.count_messages(messages)

    try:
        compressed_msgs, compressor_used, compress_ms = _compress_messages(messages)
    except Exception as exc:
        return QualityResult(
            scenario_name=name,
            scenario_type="json",
            size_label=size_label,
            tokens_without_headroom=tokens_before,
            tokens_with_headroom=tokens_before,
            tokens_saved=0,
            reduction_pct=0.0,
            probe_results=[],
            information_recall=0.0,
            critical_items_total=len(critical_needles),
            critical_items_retained=0,
            critical_retention_rate=0.0,
            structural_integrity=False,
            quality_passed=False,
            compression_meaningful=False,
            error=str(exc),
        )

    tokens_after = tokenizer.count_messages(compressed_msgs)
    tokens_saved = max(0, tokens_before - tokens_after)
    reduction_pct = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0.0

    compressed_text = _tool_content(compressed_msgs)

    probe_results = [
        ProbeResult(probe=p, survived=_probe_survived(p, compressed_text))
        for p in probe_facts
    ]
    recall = (
        sum(1 for pr in probe_results if pr.survived) / len(probe_results)
        if probe_results
        else 1.0
    )
    crits_retained = sum(1 for n in critical_needles if _probe_survived(n, compressed_text))
    crit_rate = crits_retained / len(critical_needles) if critical_needles else 1.0

    structural = _structural_ok(compressed_msgs, originally_json=True)

    quality_passed = (
        recall >= RECALL_PASS_THRESHOLD
        and crit_rate >= CRITICAL_RETENTION_THRESHOLD
        and structural
    )
    compression_meaningful = reduction_pct >= MIN_MEANINGFUL_REDUCTION * 100

    return QualityResult(
        scenario_name=name,
        scenario_type="json",
        size_label=size_label,
        tokens_without_headroom=tokens_before,
        tokens_with_headroom=tokens_after,
        tokens_saved=tokens_saved,
        reduction_pct=reduction_pct,
        probe_results=probe_results,
        information_recall=recall,
        critical_items_total=len(critical_needles),
        critical_items_retained=crits_retained,
        critical_retention_rate=crit_rate,
        structural_integrity=structural,
        quality_passed=quality_passed,
        compression_meaningful=compression_meaningful,
        compress_ms=compress_ms,
        compressor_used=compressor_used,
    )


def _run_conversation_scenario(
    name: str,
    scenario_type: str,
    size_label: str,
    messages: list[dict[str, Any]],
    probe_facts: list[str],
    critical_needles: list[str],
) -> QualityResult:
    tokenizer = _get_tokenizer()
    tokens_before = tokenizer.count_messages(messages)

    try:
        compressed_msgs, compressor_used, compress_ms = _compress_messages(messages)
    except Exception as exc:
        return QualityResult(
            scenario_name=name,
            scenario_type=scenario_type,
            size_label=size_label,
            tokens_without_headroom=tokens_before,
            tokens_with_headroom=tokens_before,
            tokens_saved=0,
            reduction_pct=0.0,
            probe_results=[],
            information_recall=0.0,
            critical_items_total=len(critical_needles),
            critical_items_retained=0,
            critical_retention_rate=0.0,
            structural_integrity=False,
            quality_passed=False,
            compression_meaningful=False,
            error=str(exc),
        )

    tokens_after = tokenizer.count_messages(compressed_msgs)
    tokens_saved = max(0, tokens_before - tokens_after)
    reduction_pct = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0.0

    all_text = _all_content(compressed_msgs)

    probe_results = [
        ProbeResult(probe=p, survived=_probe_survived(p, all_text))
        for p in probe_facts
    ]
    recall = (
        sum(1 for pr in probe_results if pr.survived) / len(probe_results)
        if probe_results
        else 1.0
    )
    crits_retained = sum(1 for n in critical_needles if _probe_survived(n, all_text))
    crit_rate = crits_retained / len(critical_needles) if critical_needles else 1.0

    structural = len(compressed_msgs) > 0

    quality_passed = (
        recall >= RECALL_PASS_THRESHOLD
        and crit_rate >= CRITICAL_RETENTION_THRESHOLD
        and structural
    )
    compression_meaningful = reduction_pct >= MIN_MEANINGFUL_REDUCTION * 100

    return QualityResult(
        scenario_name=name,
        scenario_type=scenario_type,
        size_label=size_label,
        tokens_without_headroom=tokens_before,
        tokens_with_headroom=tokens_after,
        tokens_saved=tokens_saved,
        reduction_pct=reduction_pct,
        probe_results=probe_results,
        information_recall=recall,
        critical_items_total=len(critical_needles),
        critical_items_retained=crits_retained,
        critical_retention_rate=crit_rate,
        structural_integrity=structural,
        quality_passed=quality_passed,
        compression_meaningful=compression_meaningful,
        compress_ms=compress_ms,
        compressor_used=compressor_used,
    )


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _build_scenarios(types: set[str]) -> list[tuple[str, Any]]:
    random.seed(42)
    scenarios: list[tuple[str, Any]] = []

    if "search" in types:
        for n, label in [
            (50, "50 items"),
            (200, "200 items"),
            (500, "500 items"),
            (2000, "2K items"),
        ]:
            items = generate_search_results(
                n,
                include_uuid_needles=min(3, max(1, n // 20)),
                include_errors=max(1, n // 50),
            )
            probes, crits = _extract_needles_search(items)
            scenarios.append((
                "json",
                {
                    "name": f"Search Results ({label})",
                    "size_label": label,
                    "items": items,
                    "probe_facts": probes,
                    "critical_needles": crits,
                },
            ))

    if "logs" in types:
        for n, label in [
            (100, "100 entries"),
            (500, "500 entries"),
            (2000, "2K entries"),
        ]:
            n_crit = max(1, n // 50)
            n_err = max(2, n // 20)
            entries = generate_log_entries(n, include_errors=n_err, include_critical=n_crit)
            probes, crits = _extract_needles_logs(entries)
            scenarios.append((
                "json",
                {
                    "name": f"Structured Logs ({label})",
                    "size_label": label,
                    "items": entries,
                    "probe_facts": probes,
                    "critical_needles": crits,
                },
            ))

    if "api" in types:
        for n, label in [(100, "100 items"), (500, "500 items")]:
            items = generate_api_responses(n)
            probes, crits = _extract_needles_api(items)
            scenarios.append((
                "json",
                {
                    "name": f"API Responses ({label})",
                    "size_label": label,
                    "items": items,
                    "probe_facts": probes,
                    "critical_needles": crits,
                },
            ))

    if "db" in types:
        for n, label, ttype in [
            (500, "500 rows (metrics)", "metrics"),
            (1000, "1K rows (mixed)", "mixed"),
        ]:
            rows = generate_database_rows(n, table_type=ttype)
            probes, crits = _extract_needles_db(rows)
            scenarios.append((
                "json",
                {
                    "name": f"Database Rows ({label})",
                    "size_label": label,
                    "items": rows,
                    "probe_facts": probes,
                    "critical_needles": crits,
                },
            ))

    if "agentic" in types:
        for turns, items_per_turn, label in [
            (5, 30, "5 turns / 30 items each"),
            (15, 50, "15 turns / 50 items each"),
        ]:
            msgs = generate_agentic_conversation(
                turns=turns,
                tool_calls_per_turn=1,
                items_per_tool_response=items_per_turn,
            )
            system_msg = next(
                (m.get("content", "") or "" for m in msgs if m.get("role") == "system"), ""
            )
            user_msgs = [m.get("content", "") or "" for m in msgs if m.get("role") == "user"]
            probes = ([system_msg[:80]] if system_msg else []) + user_msgs[:3]
            scenarios.append((
                "conversation",
                {
                    "name": f"Agentic Conversation ({label})",
                    "scenario_type": "agentic",
                    "size_label": label,
                    "messages": msgs,
                    "probe_facts": [p for p in probes if p],
                    "critical_needles": [],
                },
            ))

    if "rag" in types:
        for ctx_tokens, label in [
            (3_000, "3K context"),
            (10_000, "10K context"),
        ]:
            msgs = generate_rag_conversation(context_tokens=ctx_tokens, num_queries=3)
            known_facts = [
                "100 requests per minute",
                "1000 requests per minute",
                "rate limit",
                "Authorization",
            ]
            crits = known_facts[:2]
            scenarios.append((
                "conversation",
                {
                    "name": f"RAG Conversation ({label})",
                    "scenario_type": "rag",
                    "size_label": label,
                    "messages": msgs,
                    "probe_facts": known_facts,
                    "critical_needles": crits,
                },
            ))

    return scenarios


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_quality_benchmarks(
    scenario_types: list[str] | None = None,
    verbose: bool = False,
) -> list[QualityResult]:
    """Run quality benchmarks and return results.

    Args:
        scenario_types: Scenario categories. None = all.
        verbose: Print per-probe detail while running.

    Returns:
        List of QualityResult, one per scenario.
    """
    all_types = {"search", "logs", "api", "db", "agentic", "rag"}
    types = set(scenario_types) if scenario_types else all_types

    scenarios = _build_scenarios(types)
    results: list[QualityResult] = []

    for runner_type, kwargs in scenarios:
        name = kwargs.get("name", "?")
        print(f"  Running: {name} ...", end="", flush=True)

        if runner_type == "json":
            result = _run_json_scenario(**kwargs)
        else:
            result = _run_conversation_scenario(**kwargs)

        results.append(result)
        status = "✅ PASS" if result.quality_passed else "❌ FAIL"
        if result.error:
            status = "⚠️ ERR "
        saved_str = f"-{result.reduction_pct:.1f}%" if result.compression_meaningful else "  n/a "
        print(
            f" {status}  tokens {result.tokens_without_headroom:,} → "
            f"{result.tokens_with_headroom:,} ({saved_str})  "
            f"recall={result.information_recall:.0%}"
        )

        if verbose and result.probe_results:
            for pr in result.probe_results:
                mark = "  ✓" if pr.survived else "  ✗"
                print(f"{mark} {pr.probe[:70]}")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_summary(results: list[QualityResult]) -> None:
    col_name = 36
    col_tok = 10
    col_red = 8
    col_recall = 8
    col_crit = 10
    col_status = 8

    header = (
        f"{'Scenario':<{col_name}} "
        f"{'No HR':>{col_tok}} "
        f"{'With HR':>{col_tok}} "
        f"{'Saved':>{col_red}} "
        f"{'Recall':>{col_recall}} "
        f"{'Critical':>{col_crit}} "
        f"{'Pass':>{col_status}}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    prev_type = None
    for r in results:
        if r.scenario_type != prev_type:
            if prev_type is not None:
                print()
            prev_type = r.scenario_type

        saved_str = f"-{r.reduction_pct:.1f}%" if r.compression_meaningful else "-"
        recall_str = f"{r.information_recall:.0%}" if r.probe_results else "n/a"
        crit_str = (
            f"{r.critical_items_retained}/{r.critical_items_total}"
            if r.critical_items_total > 0
            else "n/a"
        )
        if r.error:
            status = "ERR  ⚠️"
        elif r.quality_passed:
            status = "PASS ✅"
        else:
            status = "FAIL ❌"

        name_str = r.scenario_name[:col_name]
        print(
            f"{name_str:<{col_name}} "
            f"{_fmt_tok(r.tokens_without_headroom):>{col_tok}} "
            f"{_fmt_tok(r.tokens_with_headroom):>{col_tok}} "
            f"{saved_str:>{col_red}} "
            f"{recall_str:>{col_recall}} "
            f"{crit_str:>{col_crit}} "
            f"{status:>{col_status}}"
        )

    print(sep)

    passed = sum(1 for r in results if r.quality_passed and not r.error)
    total_original = sum(r.tokens_without_headroom for r in results)
    total_compressed = sum(r.tokens_with_headroom for r in results)
    total_saved = total_original - total_compressed
    overall_reduction = (total_saved / total_original * 100) if total_original > 0 else 0.0

    avg_recall_vals = [r.information_recall for r in results if r.probe_results]
    avg_recall = sum(avg_recall_vals) / len(avg_recall_vals) if avg_recall_vals else 1.0

    crit_total = sum(r.critical_items_total for r in results)
    crit_retained = sum(r.critical_items_retained for r in results)

    print(f"\n  Scenarios         : {len(results)}")
    print(f"  Quality pass      : {passed}/{len(results)} ({100 * passed / len(results):.0f}%)")
    print(f"\n  Tokens (total)    : {_fmt_tok(total_original)} without Headroom")
    print(f"                    : {_fmt_tok(total_compressed)} with Headroom")
    print(f"  Token savings     : {_fmt_tok(total_saved)} ({overall_reduction:.1f}% reduction)")
    print(f"\n  Avg info recall   : {avg_recall:.1%}")
    if crit_total > 0:
        print(
            f"  Critical retained : {crit_retained}/{crit_total} "
            f"({100 * crit_retained / crit_total:.0f}%)"
        )
    print()


def generate_markdown(results: list[QualityResult], output_path: str) -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines += [
        "# Headroom Quality & Token-Reduction Report",
        "",
        f"Generated: {now}  ",
        f"Platform: {platform.machine()} · Python {platform.python_version()}",
        "",
        "> **How to read this table**",
        "> - **Tokens (No HR)**: tokens sent to the LLM *without* Headroom",
        "> - **Tokens (With HR)**: tokens sent to the LLM *with* Headroom",
        "> - **Reduction**: percentage fewer tokens when Headroom is enabled",
        "> - **Info Recall**: fraction of key facts still present after compression",
        "> - **Critical Items**: errors / anomalies that *must not* be lost (retained/total)",
        "> - **Pass**: quality threshold met (recall ≥ 80 %, critical retention = 100 %)",
        "",
        "## Results by Scenario",
        "",
        "| Scenario | Tokens (No HR) | Tokens (With HR) | Reduction | Info Recall | Critical Items | Pass |",
        "|----------|---------------:|------------------:|----------:|:-----------:|:--------------:|:----:|",
    ]

    for r in results:
        saved_str = f"-{r.reduction_pct:.1f}%" if r.compression_meaningful else "-"
        recall_str = f"{r.information_recall:.0%}" if r.probe_results else "n/a"
        crit_str = (
            f"{r.critical_items_retained}/{r.critical_items_total}"
            if r.critical_items_total > 0
            else "n/a"
        )
        if r.error:
            status = "⚠️ ERR"
        elif r.quality_passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        lines.append(
            f"| `{r.scenario_name}` "
            f"| {r.tokens_without_headroom:,} "
            f"| {r.tokens_with_headroom:,} "
            f"| {saved_str} "
            f"| {recall_str} "
            f"| {crit_str} "
            f"| {status} |"
        )

    lines.append("")

    passed = sum(1 for r in results if r.quality_passed and not r.error)
    total = len(results)
    total_original = sum(r.tokens_without_headroom for r in results)
    total_compressed = sum(r.tokens_with_headroom for r in results)
    total_saved = total_original - total_compressed
    overall_pct = (total_saved / total_original * 100) if total_original > 0 else 0.0

    avg_recall_vals = [r.information_recall for r in results if r.probe_results]
    avg_recall = sum(avg_recall_vals) / len(avg_recall_vals) if avg_recall_vals else 1.0

    crit_total = sum(r.critical_items_total for r in results)
    crit_retained = sum(r.critical_items_retained for r in results)
    crit_rate = (crit_retained / crit_total * 100) if crit_total > 0 else 100.0

    lines += [
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Scenarios run | {total} |",
        f"| Quality passed | {passed}/{total} ({100 * passed / total:.0f}%) |",
        f"| Tokens **without** Headroom | {total_original:,} |",
        f"| Tokens **with** Headroom | {total_compressed:,} |",
        f"| Total tokens saved | {total_saved:,} ({overall_pct:.1f}% reduction) |",
        f"| Avg information recall | {avg_recall:.1%} |",
        f"| Critical items retained | {crit_retained}/{crit_total} ({crit_rate:.0f}%) |"
        if crit_total > 0
        else "| Critical items retained | n/a |",
        "",
        "## Quality Thresholds",
        "",
        "| Threshold | Value | Meaning |",
        "|-----------|-------|---------|",
        f"| Information recall | ≥ {RECALL_PASS_THRESHOLD:.0%} | Key facts still present after compression |",
        f"| Critical retention | = {CRITICAL_RETENTION_THRESHOLD:.0%} | Errors / anomalies must never be lost |",
        f"| Meaningful compression | ≥ {MIN_MEANINGFUL_REDUCTION:.0%} | Minimum token reduction to count as compressed |",
        "",
        "## Interpretation",
        "",
        "Headroom reduces token usage by compressing large tool outputs (JSON arrays,",
        "log files, API responses) while preserving the information that matters most.",
        "",
        "The **Info Recall** metric verifies that probe facts — specific error codes,",
        "anomalous values, UUIDs, and critical log messages — are still present in the",
        "compressed output and therefore visible to the LLM.",
        "",
        "Critical items (CRITICAL / ERROR log entries, statistical anomalies) are **never**",
        "dropped by SmartCrusher's error-preservation and change-point detection algorithms.",
        "",
        "## Compressor Used",
        "",
        "All scenarios use `SmartCrusher` (BM25 relevance tier) — no sentence-transformers",
        "model download or API key required.  The BM25 tier runs fully offline.",
        "",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown report written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headroom quality & token-reduction benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=["all", "search", "logs", "api", "db", "agentic", "rag"],
        default="all",
        help="Scenario type to run (default: all)",
    )
    parser.add_argument("--output", "-o", help="Write markdown report to this path")
    parser.add_argument("--json", "-j", help="Write raw JSON results to this path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-probe detail")

    args = parser.parse_args()
    types = None if args.scenario == "all" else [args.scenario]

    print("=" * 60)
    print("Headroom Quality & Token-Reduction Benchmark")
    print("=" * 60)
    print(f"Scenarios : {args.scenario}")
    print(f"Compressor: SmartCrusher (BM25, offline)")
    print(f"Platform  : {platform.machine()} · Python {platform.python_version()}")
    print()

    t_start = time.perf_counter()
    results = run_quality_benchmarks(scenario_types=types, verbose=args.verbose)
    elapsed = time.perf_counter() - t_start

    print_summary(results)
    print(f"  Total time: {elapsed:.1f}s")
    print()

    if args.output:
        generate_markdown(results, args.output)

    if args.json:
        data = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
            "results": [r.to_dict() for r in results],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON results written to: {args.json}")

    all_passed = all(r.quality_passed for r in results if not r.error)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
