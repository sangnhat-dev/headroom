"""Benchmarks for tool-use safety scenarios.

Measures performance of ToolCrusher and SmartCrusher on edge-case inputs
that are common in coding-agent workflows:

- Large tool outputs with critical errors buried in the data
- Deeply nested JSON structures (schema objects, compiler diagnostics)
- Stack traces and code snippets
- Multi-step agentic conversations with many tool turns
- CCR compression-and-retrieve cycles

Run with:
    pytest benchmarks/bench_tool_use_safety.py --benchmark-only -v
"""

from __future__ import annotations

import json
import random
import string

import pytest

from headroom import RelevanceScorerConfig, SmartCrusherConfig, Tokenizer, ToolCrusherConfig
from headroom.config import CCRConfig
from headroom.transforms import ToolCrusher
from headroom.transforms.smart_crusher import SmartCrusher

random.seed(42)

# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer(mock_token_counter):
    """Tokenizer backed by the mock counter from benchmarks/conftest.py."""
    return Tokenizer(mock_token_counter, "mock-model")


@pytest.fixture
def tool_crusher():
    """ToolCrusher with aggressive settings to exercise all code paths."""
    config = ToolCrusherConfig(
        enabled=True,
        min_tokens_to_crush=0,
        max_array_items=15,
        max_string_length=500,
        max_depth=5,
    )
    return ToolCrusher(config)


@pytest.fixture
def smart_crusher():
    """SmartCrusher using BM25 (no network dependency)."""
    config = SmartCrusherConfig(
        enabled=True,
        min_tokens_to_crush=0,
        min_items_to_analyze=3,
        max_items_after_crush=15,
        preserve_change_points=True,
    )
    return SmartCrusher(config, relevance_config=RelevanceScorerConfig(tier="bm25"))


# =============================================================================
# Data generators
# =============================================================================


def make_log_items(n: int, error_rate: float = 0.05) -> list[dict]:
    """Generate log entries with occasional errors."""
    items = []
    for i in range(n):
        level = "ERROR" if random.random() < error_rate else random.choice(["INFO", "DEBUG", "WARN"])
        msg = f"Request #{i} processed in {random.randint(1, 1000)}ms" if level != "ERROR" else (
            f"Connection refused at step {i}: timeout after 30s"
        )
        items.append({
            "line": i,
            "timestamp": f"2025-01-15T12:{i // 60:02d}:{i % 60:02d}Z",
            "level": level,
            "msg": msg,
            "host": f"server-{i % 5:02d}",
        })
    return items


def make_diagnostics(n: int) -> list[dict]:
    """Generate compiler/linter diagnostics with file paths and line numbers."""
    severities = ["error", "warning", "note"]
    return [
        {
            "file": f"src/module_{i % 20}/component_{i % 10}.py",
            "line": random.randint(1, 500),
            "col": random.randint(1, 120),
            "severity": severities[i % len(severities)],
            "message": f"Undefined name 'var_{i}' in scope",
            "code": f"E{1000 + i % 100:04d}",
        }
        for i in range(n)
    ]


def make_tool_call_conversation(num_turns: int) -> list[dict]:
    """Generate a multi-turn agentic conversation with tool calls."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Analyze the repository and report all issues."},
    ]
    for turn in range(num_turns):
        call_id = f"call_{turn:04d}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "analyze_file",
                    "arguments": json.dumps({"file": f"src/module_{turn}.py"}),
                },
            }],
        })
        # Tool response with 50-item diagnostic list
        diag = make_diagnostics(50)
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(diag),
        })
    return messages


def make_stack_trace_outputs(n: int) -> list[dict]:
    """Generate JSON tool outputs containing stack traces."""
    def make_trace(i: int) -> str:
        return (
            f"Traceback (most recent call last):\n"
            f'  File "/app/worker.py", line {i * 5}, in process\n'
            f"    result = compute(data)\n"
            f'  File "/app/compute.py", line {i * 3}, in compute\n'
            f"    raise ValueError(f'Invalid input at step {i}')\n"
            f"ValueError: Invalid input at step {i}"
        )

    return [
        {
            "id": i,
            "status": "error" if i % 10 == 0 else "ok",
            "error": make_trace(i) if i % 10 == 0 else None,
            "value": i * 2.5,
        }
        for i in range(n)
    ]


# =============================================================================
# ToolCrusher benchmarks
# =============================================================================


class TestToolCrusherBenchmarks:
    """ToolCrusher performance on tool-use safety scenarios."""

    def test_compress_log_100_items(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on 100-item log output."""
        items = make_log_items(100)
        content = json.dumps({"logs": items})
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        result = benchmark(run)
        assert "<headroom:" in result.messages[0]["content"] or len(items) <= 15

    def test_compress_log_1000_items(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on 1000-item log output."""
        items = make_log_items(1000)
        content = json.dumps({"logs": items})
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_compress_diagnostics_200(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on 200-item compiler diagnostics."""
        items = make_diagnostics(200)
        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_compress_stack_traces_100(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on 100-item array with embedded stack traces."""
        items = make_stack_trace_outputs(100)
        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_compress_multi_turn_5_turns(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on a 5-turn agentic conversation."""
        messages = make_tool_call_conversation(num_turns=5)

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        result = benchmark(run)
        # All 5 tool messages should be compressed
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 5

    def test_compress_multi_turn_20_turns(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on a 20-turn agentic conversation (large context window)."""
        messages = make_tool_call_conversation(num_turns=20)

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_deeply_nested_schema_5_levels(self, benchmark, tool_crusher, mock_tokenizer):
        """ToolCrusher on a deeply nested JSON schema (5 levels)."""
        schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "array",
                            "items": {"type": "string", "enum": [f"val_{j}" for j in range(5)]},
                        }
                    },
                }
                for i in range(20)
            },
        }
        content = json.dumps(schema)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return tool_crusher.apply(messages, mock_tokenizer)

        benchmark(run)


# =============================================================================
# SmartCrusher benchmarks (coding agent safety)
# =============================================================================


class TestSmartCrusherCodingAgentBenchmarks:
    """SmartCrusher performance on coding-agent safety scenarios."""

    def test_compress_log_with_errors_100(self, benchmark, smart_crusher, mock_tokenizer):
        """SmartCrusher: 100 log items with 5% errors (must keep all errors)."""
        items = make_log_items(100, error_rate=0.05)
        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return smart_crusher.apply(messages, mock_tokenizer)

        result = benchmark(run)
        json_part = result.messages[0]["content"].split("\n<headroom:")[0]
        compressed = json.loads(json_part)
        # All error items must be preserved
        error_ids = {item["line"] for item in items if item["level"] == "ERROR"}
        compressed_ids = {item["line"] for item in compressed if isinstance(item, dict)}
        assert error_ids.issubset(compressed_ids), "All error items must be preserved"

    def test_compress_log_with_errors_500(self, benchmark, smart_crusher, mock_tokenizer):
        """SmartCrusher: 500 log items with 5% errors."""
        items = make_log_items(500, error_rate=0.05)
        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return smart_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_compress_diagnostics_100(self, benchmark, smart_crusher, mock_tokenizer):
        """SmartCrusher: 100 compiler diagnostics."""
        items = make_diagnostics(100)
        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return smart_crusher.apply(messages, mock_tokenizer)

        benchmark(run)

    def test_compress_metrics_with_spike(self, benchmark, smart_crusher, mock_tokenizer):
        """SmartCrusher: 200 metrics datapoints with anomalous spike (must be kept)."""
        items = [
            {"ts": f"T{i:04d}", "cpu": 50.0 + random.gauss(0, 2), "mem": 60.0}
            for i in range(200)
        ]
        items[120] = {"ts": "T0120", "cpu": 98.7, "mem": 99.2}  # OOM anomaly

        content = json.dumps(items)
        messages = [{"role": "tool", "tool_call_id": "c1", "content": content}]

        def run():
            return smart_crusher.apply(messages, mock_tokenizer)

        result = benchmark(run)
        json_part = result.messages[0]["content"].split("\n<headroom:")[0]
        compressed = json.loads(json_part)
        cpu_values = [item["cpu"] for item in compressed if isinstance(item, dict) and "cpu" in item]
        assert 98.7 in cpu_values, "CPU spike must be preserved"

    def test_plain_text_passthrough_1000_chars(self, benchmark, smart_crusher, mock_tokenizer):
        """SmartCrusher must not modify plain-text code output (fast passthrough)."""
        code = (
            "#!/usr/bin/env python3\n"
            "# Auto-generated by build system\n\n"
            + ("x = 1\n" * 200)  # ~1000 chars of code
        )
        messages = [{"role": "tool", "tool_call_id": "c1", "content": code}]

        def run():
            return smart_crusher.apply(messages, mock_tokenizer)

        result = benchmark(run)
        assert result.messages[0]["content"] == code, "Plain text must be unchanged"
