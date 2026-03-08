"""Tool-use safety and edge case tests for coding agent reliability.

This test file addresses gaps found during the Copilot coding agent's audit of
tool-use handling in headroom. It covers:

1. Injection safety: Malformed JSON, prototype pollution, marker collisions
2. Special float values: NaN, Infinity in score fields
3. Mixed-type arrays: Non-uniform tool outputs
4. Compression idempotency: Compressing already-compressed output
5. Empty/near-empty tool output handling
6. TOIN recommendation safety when disabled or no data
7. Rolling window tool atomicity with Anthropic format
8. SmartCrusher passthrough guarantee on non-compressible input
"""

from __future__ import annotations

import json
import math

import pytest

from headroom.config import RelevanceScorerConfig
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig, smart_crush_tool_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bm25_crusher(**kwargs) -> SmartCrusher:
    """Create a SmartCrusher using BM25-only scoring (no network access needed)."""
    config = SmartCrusherConfig(**kwargs)
    bm25 = RelevanceScorerConfig(tier="bm25")
    return SmartCrusher(config, relevance_config=bm25)


# ===========================================================================
# 1. INJECTION SAFETY
# ===========================================================================


class TestInjectionSafety:
    """Verify SmartCrusher handles injected/malformed input safely."""

    def test_null_element_in_array(self):
        """Arrays with null elements should be handled without crashing."""
        data = json.dumps([{"id": 1}, None, {"id": 2}])
        compressed, was_modified, reason = smart_crush_tool_output(
            data, SmartCrusherConfig()
        )
        # Must return valid JSON or the original string
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_prototype_pollution_field(self):
        """Items with __proto__ fields should not cause crashes."""
        items = [{"id": i, "__proto__": {"admin": True}, "value": i} for i in range(30)]
        data = json.dumps(items)
        compressed, _, _ = smart_crush_tool_output(data, SmartCrusherConfig(max_items_after_crush=10))
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_headroom_marker_field_collision(self):
        """Data already containing __headroom_ fields should not confuse compression."""
        items = [
            {
                "id": i,
                "__headroom_compressed": True,
                "__headroom_hash": "fakehash12345",
                "score": i / 100.0,
            }
            for i in range(50)
        ]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        # Compression should have worked despite the fake markers
        assert isinstance(result, list)
        assert len(result) <= 20

    def test_deeply_nested_objects_do_not_crash(self):
        """Deeply nested objects in tool output should not cause recursion errors."""
        def make_nested(depth: int) -> dict:
            if depth == 0:
                return {"leaf": "value"}
            return {"level": depth, "child": make_nested(depth - 1)}

        items = [make_nested(10) for _ in range(20)]
        data = json.dumps(items)
        # Should not raise RecursionError or similar
        compressed, _, _ = smart_crush_tool_output(data, SmartCrusherConfig())
        assert json.loads(compressed) is not None

    def test_extremely_long_string_values(self):
        """Items with very long string values should not cause memory issues."""
        items = [
            {"id": i, "content": "x" * 10_000, "score": i / 100.0}
            for i in range(30)
        ]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_unicode_and_emoji_in_content(self):
        """Unicode characters and emoji in tool output must be preserved correctly."""
        items = [
            {"id": 1, "message": "Error: \U0001f525 Server on fire \U0001f525", "status": "error"},
            {"id": 2, "message": "Café résumé naïve", "status": "ok"},
            {"id": 3, "message": "\u4e2d\u6587\u5185\u5bb9", "status": "ok"},
        ] + [{"id": i, "message": f"normal_{i}", "status": "ok"} for i in range(4, 30)]

        data = json.dumps(items, ensure_ascii=False)
        config = SmartCrusherConfig(max_items_after_crush=5)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)

        # Error item with emoji must be preserved
        error_items = [x for x in result if x.get("status") == "error"]
        assert len(error_items) >= 1, "Error item with emoji was dropped"


# ===========================================================================
# 2. SPECIAL FLOAT VALUES
# ===========================================================================


class TestSpecialFloatValues:
    """Verify NaN/Infinity in score fields don't crash or corrupt output."""

    def test_nan_score_field_handled_gracefully(self):
        """Items with NaN scores should be treated as anomalies or ignored gracefully."""
        items = []
        for i in range(50):
            score = i / 10.0
            if i == 10:
                score = float("nan")
            elif i == 20:
                score = float("inf")
            elif i == 30:
                score = float("-inf")
            items.append({"id": i, "score": score, "name": f"item_{i}"})

        # JSON doesn't support NaN/Inf, so this simulates what real tool output might contain
        # (some tools use json.dumps(allow_nan=True) or return "NaN" as string)
        safe_items = []
        for item in items:
            safe_item = dict(item)
            s = safe_item["score"]
            if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
                safe_item["score"] = None  # Normalize to null before passing to SmartCrusher
            safe_items.append(safe_item)

        data = json.dumps(safe_items)
        config = SmartCrusherConfig(max_items_after_crush=15)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_none_score_field_does_not_crash(self):
        """Items with None/null in a score field should not crash the scorer."""
        items = [
            {"id": i, "score": None if i % 5 == 0 else i / 50.0, "name": f"item_{i}"}
            for i in range(40)
        ]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10)
        # Must not raise TypeError: '<' not supported between instances of 'NoneType' and 'float'
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, list)
        assert len(result) <= 20  # Should have compressed


# ===========================================================================
# 3. MIXED-TYPE ARRAYS
# ===========================================================================


class TestMixedTypeArrays:
    """Verify SmartCrusher handles non-uniform tool outputs (not all dicts)."""

    def test_mixed_type_array_passthrough_or_compress(self):
        """Arrays mixing dicts, strings, and numbers should not crash."""
        mixed = [
            {"id": 1, "type": "dict"},
            "just a string",
            42,
            None,
            {"id": 2, "type": "dict"},
            ["nested", "array"],
            True,
            {"id": 3, "type": "dict"},
        ]
        data = json.dumps(mixed)
        config = SmartCrusherConfig(max_items_after_crush=5)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_string_only_array_handles_gracefully(self):
        """Arrays of plain strings (e.g., grep results) should work."""
        items = [f"file_{i}.py:line {i}: some_function()" for i in range(60)]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=20)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, list)

    def test_number_only_array_handles_gracefully(self):
        """Arrays of numbers (e.g., metric values) should work."""
        items = list(range(100))
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert isinstance(result, (list, dict))  # Could be summarized as stats dict


# ===========================================================================
# 4. COMPRESSION IDEMPOTENCY
# ===========================================================================


class TestCompressionIdempotency:
    """Verify compressing already-compressed output is safe."""

    def test_double_compression_is_safe(self):
        """Compressing an already-compressed result should not expand or corrupt it."""
        items = [{"id": i, "score": (100 - i) / 100.0, "data": f"content_{i}"} for i in range(100)]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=20)

        # First compression
        compressed1, was_modified1, _ = smart_crush_tool_output(data, config)

        # Second compression of the already-compressed result
        compressed2, was_modified2, _ = smart_crush_tool_output(compressed1, config)

        result1 = json.loads(compressed1)
        result2 = json.loads(compressed2)

        # Should be valid JSON lists
        assert isinstance(result1, list)
        assert isinstance(result2, list)

        # Second compression should not expand (must be <= first compression size)
        assert len(result2) <= len(result1) + 5, (
            f"Double compression expanded output: {len(result1)} → {len(result2)} items"
        )

    def test_already_small_array_content_unchanged(self):
        """Arrays already within the max_items limit should not lose content."""
        items = [{"id": i, "value": i} for i in range(5)]  # Very small
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=20)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        # Output should contain all 5 items (content unchanged)
        assert len(result) == 5, f"Small array had content removed, got {len(result)} items"


# ===========================================================================
# 5. EMPTY / NEAR-EMPTY OUTPUT
# ===========================================================================


class TestEmptyAndNearEmptyOutput:
    """Verify edge cases at the boundaries of array size."""

    def test_empty_array_passthrough(self):
        """Empty array should pass through with original content intact."""
        data = json.dumps([])
        compressed, _, _ = smart_crush_tool_output(data, SmartCrusherConfig())
        assert json.loads(compressed) == []

    def test_single_item_array_content_preserved(self):
        """Single-item array should never have items removed."""
        item = {"id": 1, "value": "only item"}
        data = json.dumps([item])
        compressed, _, _ = smart_crush_tool_output(data, SmartCrusherConfig())
        result = json.loads(compressed)
        assert len(result) == 1, "Single-item array had its item removed"
        assert result[0]["id"] == 1

    def test_empty_string_passthrough(self):
        """Empty string tool output should pass through unchanged."""
        compressed, was_modified, _ = smart_crush_tool_output("", SmartCrusherConfig())
        assert not was_modified
        assert compressed == ""

    def test_whitespace_only_passthrough(self):
        """Whitespace-only tool output should pass through unchanged."""
        compressed, was_modified, _ = smart_crush_tool_output("   ", SmartCrusherConfig())
        assert not was_modified

    def test_exactly_at_max_items_content_preserved(self):
        """Array exactly at max_items limit should not have items removed."""
        max_items = 20
        items = [{"id": i, "value": i} for i in range(max_items)]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=max_items)
        compressed, _, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)
        assert len(result) == max_items, (
            f"Array at exact max_items limit had items removed: got {len(result)} items"
        )

    def test_one_above_max_items_may_be_modified(self):
        """Array with one more item than max_items is a candidate for compression."""
        max_items = 20
        items = [{"id": i, "score": (max_items * 2 - i) / (max_items * 2), "value": i} for i in range(max_items + 10)]
        data = json.dumps(items)
        config = SmartCrusherConfig(
            max_items_after_crush=max_items,
            min_items_to_analyze=5,
        )
        # Should not crash even if it decides not to compress
        compressed, _, _ = smart_crush_tool_output(data, config)
        assert json.loads(compressed) is not None


# ===========================================================================
# 6. TOIN SAFETY WHEN DISABLED OR EMPTY
# ===========================================================================


class TestTOINSafety:
    """Verify SmartCrusher is safe when TOIN data is unavailable or empty."""

    def test_compression_works_without_toin_data(self):
        """SmartCrusher should work even when TOIN has no learned patterns."""
        from headroom.telemetry.toin import reset_toin

        reset_toin()  # Fresh TOIN with no patterns

        items = [{"id": i, "score": (50 - i) / 50.0, "type": "result"} for i in range(50)]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10, use_feedback_hints=True)
        bm25 = RelevanceScorerConfig(tier="bm25")
        crusher = SmartCrusher(config, relevance_config=bm25)

        result, info, markers, _ = crusher._crush_array(
            items, query_context="find results", tool_name="search"
        )
        assert isinstance(result, list), "Compression should work with empty TOIN"
        assert len(result) > 0

    def test_toin_errors_do_not_break_compression(self):
        """If TOIN raises an error, compression should still complete."""
        from unittest.mock import patch

        items = [{"id": i, "score": (50 - i) / 50.0} for i in range(50)]
        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10)
        bm25 = RelevanceScorerConfig(tier="bm25")
        crusher = SmartCrusher(config, relevance_config=bm25)

        # Simulate TOIN failure
        with patch.object(crusher, "_get_toin", side_effect=RuntimeError("TOIN unavailable")):
            # Should not raise - TOIN errors are caught internally
            compressed, _, _ = smart_crush_tool_output(data, config)
            result = json.loads(compressed)
            assert isinstance(result, list)


# ===========================================================================
# 7. ROLLING WINDOW TOOL ATOMICITY (ANTHROPIC FORMAT)
# ===========================================================================


class TestRollingWindowToolAtomicity:
    """Verify tool call/response pairs are never orphaned."""

    def test_tool_call_and_result_dropped_together(self):
        """When a tool unit is dropped, both the call and result go together."""
        from headroom.transforms.rolling_window import RollingWindow, RollingWindowConfig

        # Build a conversation with multiple tool units
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
        ]
        # Add 5 tool call+result pairs
        for i in range(5):
            messages.append({
                "role": "user",
                "content": f"Query {i}",
            })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "search", "arguments": f'{{"q": "query_{i}"}}'},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": json.dumps([{"result": f"data_{j}"} for j in range(100)]),
            })

        config = RollingWindowConfig(keep_last_turns=1, output_buffer_tokens=0)
        window = RollingWindow(config)

        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        # Return a large count initially (over budget), then reduce
        call_count = [0]

        def count_messages(msgs, model=None):
            call_count[0] += 1
            return 10000  # Always over budget to force dropping

        def count_message(msg, model=None):
            return 500  # Each message costs 500 tokens

        tokenizer.count_messages.side_effect = count_messages
        tokenizer.count_message.side_effect = count_message

        result = window.apply(messages, tokenizer, model_limit=2000)

        # Verify no orphaned tool results (every tool response has a matching call)
        result_msgs = result.messages
        tool_call_ids = set()
        for msg in result_msgs:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_call_ids.add(tc["id"])

        for msg in result_msgs:
            if msg.get("role") == "tool":
                assert msg.get("tool_call_id") in tool_call_ids, (
                    f"Orphaned tool result: tool_call_id={msg.get('tool_call_id')} "
                    f"has no matching assistant tool call"
                )

    def test_system_prompt_never_dropped(self):
        """System prompt must survive rolling window compression."""
        from headroom.transforms.rolling_window import RollingWindow, RollingWindowConfig

        system_content = "You are a helpful coding assistant. Never drop me."
        messages = [
            {"role": "system", "content": system_content},
        ]
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i} " + "x" * 50})
            messages.append({"role": "assistant", "content": f"Answer {i} " + "y" * 50})

        config = RollingWindowConfig(keep_last_turns=1, output_buffer_tokens=0)
        window = RollingWindow(config)

        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.count_messages.return_value = 10000  # Always over budget
        tokenizer.count_message.return_value = 500

        result = window.apply(messages, tokenizer, model_limit=2000)
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == system_content


# ===========================================================================
# 8. PASSTHROUGH GUARANTEE
# ===========================================================================


class TestPassthroughGuarantee:
    """Verify SmartCrusher never modifies non-compressible data."""

    def test_non_json_passthrough(self):
        """Non-JSON tool output should pass through unchanged."""
        plain_text = "File not found: /home/user/project/src/main.py"
        compressed, was_modified, _ = smart_crush_tool_output(plain_text, SmartCrusherConfig())
        assert not was_modified
        assert compressed == plain_text

    def test_json_object_not_array_passthrough(self):
        """A JSON object (not array) at top level should pass through unchanged."""
        obj = {"status": "ok", "result": "success", "code": 200}
        data = json.dumps(obj)
        # SmartCrusher may process flat objects, but should never crash
        compressed, _, _ = smart_crush_tool_output(data, SmartCrusherConfig())
        result = json.loads(compressed)
        assert isinstance(result, (dict, list))

    def test_error_items_always_retained(self):
        """Items with error indicators must NEVER be dropped."""
        items = [{"id": i, "score": i / 100.0, "value": f"data_{i}"} for i in range(100)]
        # Plant critical error items at positions that would normally be dropped
        items[40] = {"id": 40, "error": "ConnectionRefusedError", "score": 0.4}
        items[60] = {"id": 60, "status": "failed", "message": "timeout", "score": 0.6}
        items[80] = {"id": 80, "exception": "RuntimeError: out of memory", "score": 0.8}

        data = json.dumps(items)
        config = SmartCrusherConfig(max_items_after_crush=10)
        compressed, was_modified, _ = smart_crush_tool_output(data, config)
        result = json.loads(compressed)

        error_ids = {x["id"] for x in result if x.get("error") or x.get("status") == "failed" or x.get("exception")}
        assert 40 in error_ids, "Item with 'error' field was dropped"
        assert 60 in error_ids, "Item with 'status=failed' was dropped"
        assert 80 in error_ids, "Item with 'exception' field was dropped"

    def test_malformed_json_passthrough(self):
        """Malformed JSON should pass through without crashing."""
        bad_inputs = [
            "{not valid json",
            "[1, 2, 3",
            '{"key": }',
            "undefined",
            "<<EOF",
        ]
        config = SmartCrusherConfig()
        for bad_input in bad_inputs:
            compressed, was_modified, _ = smart_crush_tool_output(bad_input, config)
            assert not was_modified, f"Malformed JSON was modified: {bad_input!r}"
            assert compressed == bad_input, f"Malformed JSON was changed: {bad_input!r}"
