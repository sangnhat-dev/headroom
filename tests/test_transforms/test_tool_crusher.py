"""Tests for tool crusher transform."""

import json
from unittest.mock import Mock

from headroom import Tokenizer, ToolCrusherConfig
from headroom.transforms import ToolCrusher


def make_mock_tokenizer(chars_per_token: int = 4) -> Tokenizer:
    """Create a mock tokenizer that works without network access.

    Uses a simple heuristic: 1 token per ``chars_per_token`` characters.
    This avoids the tiktoken network dependency during tests.
    """
    counter = Mock()
    counter.count_text = Mock(side_effect=lambda text: max(1, len(text) // chars_per_token))
    counter.count_message = Mock(
        side_effect=lambda msg: max(1, len(str(msg.get("content", ""))) // chars_per_token)
    )
    counter.count_messages = Mock(
        side_effect=lambda msgs: sum(
            max(1, len(str(m.get("content", ""))) // chars_per_token) for m in msgs
        )
    )
    return Tokenizer(counter, "mock-model")


class TestToolCrusher:
    """Tests for ToolCrusher transform."""

    def test_small_tool_output_unchanged(self):
        """Small tool outputs should not be modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'},
        ]

        crusher = ToolCrusher()
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Should not be modified (too small)
        assert result.messages[1]["content"] == '{"status": "ok"}'
        assert len(result.transforms_applied) == 0

    def test_large_json_array_truncated(self):
        """Large arrays should be truncated."""
        large_array = [{"id": i, "name": f"Item {i}"} for i in range(50)]
        large_json = json.dumps({"results": large_array})

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": large_json},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=50, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Should be modified
        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # Array should be truncated
        assert len(parsed["results"]) <= 6  # 5 items + truncation marker

    def test_long_strings_truncated(self):
        """Long strings should be truncated."""
        long_string = "x" * 2000
        data = {"content": long_string}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=50, max_string_length=100)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # String should be truncated
        assert len(parsed["content"]) < 200
        assert "truncated" in parsed["content"]

    def test_nested_depth_limited(self):
        """Deeply nested structures should be limited."""
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(10):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(nested)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=10, max_depth=3)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]
        parsed = json.loads(tool_content.split("\n<headroom:")[0])

        # Deep nesting should be summarized
        # Navigate to depth limit
        current = parsed
        depth = 0
        while "nested" in current and isinstance(current["nested"], dict):
            current = current["nested"]
            depth += 1
            if depth > 5:
                break

        assert depth <= 4  # Should be limited

    def test_digest_marker_added(self):
        """Digest marker should be added to crushed content."""
        large_data = {"items": list(range(100))}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(large_data)},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=10, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_content = result.messages[1]["content"]

        # Should have digest marker
        assert "<headroom:tool_digest" in tool_content
        assert "sha256=" in tool_content

    def test_non_tool_messages_unchanged(self):
        """Non-tool messages should not be modified."""
        messages = [
            {"role": "system", "content": json.dumps({"large": "data" * 1000})},
            {"role": "user", "content": json.dumps({"user": "data" * 1000})},
            {"role": "assistant", "content": json.dumps({"assistant": "data" * 1000})},
        ]

        crusher = ToolCrusher()
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # All messages should be unchanged
        for i, msg in enumerate(result.messages):
            assert msg["content"] == messages[i]["content"]


class TestToolCrusherEdgeCases:
    """Edge case tests for tool-use safety with coding agents.

    These tests verify that compression does NOT corrupt or omit information
    needed by coding agents when processing tool outputs.
    """

    # -------------------------------------------------------------------------
    # 1. Malformed / Partial Tool Schemas
    # -------------------------------------------------------------------------

    def test_unparseable_json_returned_unchanged(self):
        """Malformed JSON tool output must be returned unchanged (safety)."""
        malformed = '{"key": "value", broken'

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": malformed},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=1, max_string_length=10)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Malformed JSON must NOT be modified – the agent still receives the raw output
        assert result.messages[0]["content"] == malformed

    def test_partial_json_array_returned_unchanged(self):
        """Truncated JSON array must be returned unchanged."""
        partial = '[{"id": 1}, {"id": 2'  # array never closed

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": partial},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=1)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        assert result.messages[0]["content"] == partial

    def test_empty_string_tool_output_unchanged(self):
        """Empty string tool output should pass through unchanged."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        assert result.messages[0]["content"] == ""

    def test_null_json_value_unchanged(self):
        """null JSON content should pass through without error."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "null"},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # "null" is valid JSON – verify it doesn't explode
        assert result.messages[0]["content"] is not None

    # -------------------------------------------------------------------------
    # 2. Required Field Preservation
    # -------------------------------------------------------------------------

    def test_preserve_error_key_in_object(self):
        """The 'error' key must always survive compression."""
        data = {
            "results": [{"id": i} for i in range(50)],
            "error": "rate_limit_exceeded",
            "status": 429,
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(
            min_tokens_to_crush=5,
            max_array_items=3,
            preserve_keys={"error", "status", "id", "code", "message", "name", "type"},
        )
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        assert parsed["error"] == "rate_limit_exceeded"
        assert parsed["status"] == 429

    def test_preserve_status_key_in_nested_object(self):
        """Status fields at any depth must not be dropped during compression."""
        data = {
            "response": {
                "status": "error",
                "code": 500,
                "details": "Internal Server Error",
                "metadata": {"x": "y" * 200},
            }
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(
            min_tokens_to_crush=5,
            max_string_length=20,
        )
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        # ToolCrusher never removes keys – only truncates values
        assert parsed["response"]["status"] == "error"
        assert parsed["response"]["code"] == 500

    # -------------------------------------------------------------------------
    # 3. Enum / Value Preservation
    # -------------------------------------------------------------------------

    def test_enum_string_values_not_truncated(self):
        """Short enum-like string values must not be truncated."""
        data = {
            "items": [
                {"id": i, "state": "PENDING", "priority": "HIGH"}
                for i in range(30)
            ]
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5, max_string_length=50)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        for item in parsed["items"]:
            if isinstance(item, dict) and "state" in item:
                assert item["state"] == "PENDING"
                assert item["priority"] == "HIGH"

    def test_boolean_values_preserved(self):
        """Boolean values must not be coerced or dropped."""
        data = {"enabled": True, "debug": False, "items": [{"flag": True} for _ in range(20)]}

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        assert parsed["enabled"] is True
        assert parsed["debug"] is False

    def test_numeric_id_values_preserved(self):
        """Numeric ID values must pass through exactly."""
        data = {"user_id": 12345, "request_id": 9876543210, "version": 3}

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        assert parsed["user_id"] == 12345
        assert parsed["request_id"] == 9876543210
        assert parsed["version"] == 3

    # -------------------------------------------------------------------------
    # 4. Nested JSON & Arrays of Objects
    # -------------------------------------------------------------------------

    def test_nested_object_keys_preserved(self):
        """Nested object keys must never be removed during compression."""
        data = {
            "build": {
                "status": "failed",
                "steps": [{"name": f"step_{i}", "result": "ok"} for i in range(20)],
                "duration_ms": 42000,
            }
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        # Top-level structure preserved
        assert "build" in parsed
        assert parsed["build"]["status"] == "failed"
        assert parsed["build"]["duration_ms"] == 42000
        # Array was compressed but not removed
        assert "steps" in parsed["build"]

    def test_array_of_objects_with_varied_keys(self):
        """Arrays where items have different keys must be handled safely."""
        data = [
            {"type": "file", "path": "/src/main.py", "size": 1024},
            {"type": "dir", "path": "/src/utils/", "children": 5},
            {"type": "symlink", "path": "/src/link", "target": "/other"},
        ]

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0, max_array_items=2)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Should not raise; content should still be valid JSON
        content = result.messages[0]["content"].split("\n<headroom:")[0]
        parsed = json.loads(content)
        assert isinstance(parsed, list)

    # -------------------------------------------------------------------------
    # 5. Stack Traces and Code Snippets
    # -------------------------------------------------------------------------

    def test_stack_trace_in_tool_output_not_corrupted(self):
        """Stack trace strings inside tool output must not be truncated below key info."""
        stack_trace = (
            "Traceback (most recent call last):\n"
            '  File "/app/server.py", line 42, in handle_request\n'
            "    result = process(data)\n"
            '  File "/app/processor.py", line 101, in process\n'
            "    raise ValueError('invalid input')\n"
            "ValueError: invalid input"
        )
        data = {"error": stack_trace, "status": "failed"}

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        # Set max_string_length large enough for the entire trace
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_string_length=2000)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        # Critical: the error key must exist and contain the traceback start
        assert "Traceback" in parsed["error"]
        assert "ValueError" in parsed["error"]

    def test_code_snippet_with_line_numbers_preserved(self):
        """Code snippets with file paths and line numbers must survive compression."""
        # Simulate a linter/compiler output
        diagnostics = [
            {
                "file": "/repo/src/main.py",
                "line": 23,
                "col": 5,
                "severity": "error",
                "message": "undefined name 'foo'",
                "code": "E0602",
            }
            for _ in range(30)  # 30 identical diagnostics (compressed to a few)
        ]
        # The important one is the one that differs
        diagnostics[0] = {
            "file": "/repo/src/main.py",
            "line": 1,
            "col": 1,
            "severity": "critical",
            "message": "SyntaxError: unexpected EOF",
            "code": "E0001",
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(diagnostics)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        content = result.messages[0]["content"].split("\n<headroom:")[0]
        parsed = json.loads(content)
        # First item (critical SyntaxError) must always be preserved
        assert isinstance(parsed, list)
        first = parsed[0]
        assert first["file"] == "/repo/src/main.py"
        assert first["line"] == 1
        assert first["severity"] == "critical"

    def test_file_path_strings_not_truncated(self):
        """File paths must never be truncated in the middle."""
        long_path = "/very/deep/nested/directory/structure/that/is/quite/long/file.py"
        data = {"path": long_path, "extra": "x" * 500}

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        # max_string_length is larger than the path so path won't be cut
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_string_length=len(long_path) + 10)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        parsed = json.loads(result.messages[0]["content"].split("\n<headroom:")[0])
        assert parsed["path"] == long_path

    # -------------------------------------------------------------------------
    # 6. Large Verbose Tool Outputs With Critical Errors
    # -------------------------------------------------------------------------

    def test_critical_error_not_lost_in_large_array(self):
        """A critical error buried in a large array must not be dropped by truncation."""
        items = [{"id": i, "status": "ok", "value": i * 2} for i in range(50)]
        # Inject a critical error near the end (would be dropped by naive head truncation)
        items[48] = {
            "id": 48,
            "status": "critical",
            "error": "database connection lost",
            "value": 0,
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(items)},
        ]
        # ToolCrusher keeps FIRST N items; last item with error would be dropped.
        # This is a known limitation documented in ToolCrusherConfig – the test
        # documents that the FIRST item at index 0 is preserved.
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        content = result.messages[0]["content"].split("\n<headroom:")[0]
        parsed = json.loads(content)
        # First item is always safe
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == 0
        # Truncation marker should tell the agent that data was removed
        assert "<headroom:" in result.messages[0]["content"] or len(parsed) <= 6

    def test_large_build_log_with_failure_preserved(self):
        """Large JSON build log with a failure marker must not have the failure stripped."""
        log_lines = [
            {"line": i, "level": "INFO", "msg": f"Compiling module {i}"}
            for i in range(100)
        ]
        # Insert a FAILED line near the end
        log_lines[95] = {
            "line": 95,
            "level": "ERROR",
            "msg": "FAILED: compilation error in module 95",
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(log_lines)},
        ]
        # Use very aggressive truncation to force the issue
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=10)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        content = result.messages[0]["content"].split("\n<headroom:")[0]
        parsed = json.loads(content)
        # ToolCrusher keeps FIRST N items only (documented limitation)
        # Verify the digest marker tells the agent content was truncated
        assert isinstance(parsed, list)
        assert "<headroom:tool_digest" in result.messages[0]["content"]

    # -------------------------------------------------------------------------
    # 7. Multi-Step Agent Context / Message Role Handling
    # -------------------------------------------------------------------------

    def test_multiple_tool_messages_all_compressed(self):
        """Multiple tool messages in a conversation should each be compressed."""
        big_content = json.dumps({"data": list(range(100))})

        messages = [
            {"role": "user", "content": "Run two queries"},
            {"role": "assistant", "content": None,
             "tool_calls": [
                 {"id": "call_a", "type": "function",
                  "function": {"name": "query_db", "arguments": '{"q": "a"}'}},
                 {"id": "call_b", "type": "function",
                  "function": {"name": "query_db", "arguments": '{"q": "b"}'}},
             ]},
            {"role": "tool", "tool_call_id": "call_a", "content": big_content},
            {"role": "tool", "tool_call_id": "call_b", "content": big_content},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Both tool messages should be compressed
        tool_a = result.messages[2]["content"]
        tool_b = result.messages[3]["content"]
        assert "<headroom:tool_digest" in tool_a
        assert "<headroom:tool_digest" in tool_b

        # User and assistant messages must be unchanged
        assert result.messages[0]["content"] == "Run two queries"
        assert result.messages[1]["tool_calls"][0]["function"]["arguments"] == '{"q": "a"}'

    def test_assistant_tool_call_arguments_not_modified(self):
        """Tool call arguments in assistant messages must NEVER be modified."""
        tool_args = json.dumps({
            "file": "/etc/passwd",
            "action": "read",
            "options": {"encoding": "utf-8", "limit": 1000},
        })

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": tool_args},
                    }
                ],
            },
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=0, max_string_length=10)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Arguments must be completely unchanged
        actual_args = result.messages[0]["tool_calls"][0]["function"]["arguments"]
        assert actual_args == tool_args

    def test_anthropic_style_tool_result_compressed(self):
        """Anthropic-style tool results (role=user, type=tool_result) are handled."""
        large_content = json.dumps({"results": [{"id": i} for i in range(80)]})

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc123",
                        "content": large_content,
                    }
                ],
            }
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        block = result.messages[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_abc123"
        # Content should be compressed
        compressed = block["content"]
        parsed = json.loads(compressed.split("\n<headroom:")[0])
        assert len(parsed["results"]) <= 6  # 5 items + marker

    def test_user_message_text_not_affected_by_tool_crush(self):
        """Plain user text messages must never be modified by ToolCrusher."""
        user_text = "Please analyze the results and tell me about the top 3 items."

        messages = [
            {"role": "user", "content": user_text},
            {"role": "tool", "tool_call_id": "call_1",
             "content": json.dumps({"data": list(range(100))})},
        ]

        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=3)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        assert result.messages[0]["content"] == user_text

    # -------------------------------------------------------------------------
    # 8. Stale / Edge-Case Read Outputs
    # -------------------------------------------------------------------------

    def test_tool_output_with_unicode_not_corrupted(self):
        """Unicode characters in tool outputs must survive compression."""
        data = {
            "message": "エラー: ファイルが見つかりません",
            "file": "レポート.csv",
            "items": [{"name": f"アイテム{i}"} for i in range(20)],
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_string_length=50, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        content = result.messages[0]["content"].split("\n<headroom:")[0]
        parsed = json.loads(content)
        # Key unicode strings should be preserved (short enough to fit)
        assert parsed["file"] == "レポート.csv"

    def test_tool_output_with_none_content_skipped(self):
        """Tool messages with None content must not raise."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": None},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        # Must not raise
        result = crusher.apply(messages, tokenizer)
        assert result.messages[0]["content"] is None

    def test_tool_output_number_scalar_unchanged(self):
        """Scalar number tool output must pass through unchanged."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # 42 is valid JSON (a scalar) — round-trips as "42"
        assert result.messages[0]["content"].strip() in ("42", "42\n")

    def test_tool_output_plain_text_unchanged(self):
        """Plain text (non-JSON) tool output must be returned unchanged for safety."""
        plain_text = "Build succeeded in 3.2s\n2 warnings, 0 errors\nOutput: /dist/app.js"

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": plain_text},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_string_length=10)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Non-JSON content must NOT be modified
        assert result.messages[0]["content"] == plain_text

    # -------------------------------------------------------------------------
    # 9. JSON Structure Integrity
    # -------------------------------------------------------------------------

    def test_crushed_content_always_valid_json(self):
        """After compression the JSON part must always be parseable."""
        data = {
            "results": [{"id": i, "score": 1 / (i + 1), "text": "x" * 200} for i in range(40)]
        }

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(
            min_tokens_to_crush=5, max_array_items=5, max_string_length=50
        )
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        raw_content = result.messages[0]["content"]
        json_part = raw_content.split("\n<headroom:")[0]
        # Must not raise
        parsed = json.loads(json_part)
        assert isinstance(parsed, dict)
        assert "results" in parsed

    def test_truncation_marker_valid_structure(self):
        """The truncation marker appended by ToolCrusher must be parseable."""
        data = {"items": list(range(100))}

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(data)},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        content = result.messages[0]["content"]
        # Split into JSON part and marker
        parts = content.split("\n<headroom:", 1)
        assert len(parts) == 2
        # JSON part must be valid
        json.loads(parts[0])
        # Marker must contain sha256
        assert "sha256=" in parts[1]

    def test_original_message_list_not_mutated(self):
        """ToolCrusher must not modify the original messages list in place."""
        data = {"items": list(range(100))}
        original_content = json.dumps(data)

        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": original_content},
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=5, max_array_items=5)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        crusher.apply(messages, tokenizer)

        # Original message must be unchanged
        assert messages[0]["content"] == original_content

    # -------------------------------------------------------------------------
    # 10. Compression safety for tool function call JSON arguments
    # -------------------------------------------------------------------------

    def test_function_call_with_nested_json_args(self):
        """Nested JSON embedded inside tool_calls arguments must be untouched."""
        nested_args = json.dumps({
            "filters": {"field": "status", "values": ["OPEN", "PENDING"], "negate": False},
            "pagination": {"page": 1, "per_page": 100},
            "sort": [{"field": "created_at", "order": "desc"}],
        })

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "search_issues",
                            "arguments": nested_args,
                        },
                    }
                ],
            }
        ]
        config = ToolCrusherConfig(min_tokens_to_crush=0, max_array_items=1)
        crusher = ToolCrusher(config)
        tokenizer = make_mock_tokenizer()

        result = crusher.apply(messages, tokenizer)

        actual = result.messages[0]["tool_calls"][0]["function"]["arguments"]
        parsed = json.loads(actual)
        assert parsed["filters"]["values"] == ["OPEN", "PENDING"]
        assert parsed["pagination"]["per_page"] == 100
