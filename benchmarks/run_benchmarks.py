#!/usr/bin/env python3
"""CLI runner for Headroom benchmark suite.

This script provides a convenient interface for running benchmarks and
generating reports. It wraps pytest-benchmark with Headroom-specific
options and markdown report generation.

Usage:
    # Run all benchmarks
    python benchmarks/run_benchmarks.py

    # Run specific suite
    python benchmarks/run_benchmarks.py --suite transforms

    # Generate markdown report
    python benchmarks/run_benchmarks.py --output report.md

    # Compare against baseline
    python benchmarks/run_benchmarks.py --compare baseline.json

    # Save results as new baseline
    python benchmarks/run_benchmarks.py --save-baseline baseline.json

    # Quality & token-reduction benchmark (no API key required)
    python benchmarks/run_benchmarks.py --suite quality --output benchmarks/QUALITY_RESULTS.md

Available Suites:
    all         - Run all benchmark suites (transforms + relevance + tool-safety)
    latency     - Compression overhead & cost-benefit analysis (standalone)
    quality     - Output quality & token reduction with/without Headroom (standalone)
    transforms  - SmartCrusher, CacheAligner, RollingWindow
    relevance   - BM25Scorer, HybridScorer
    tool-safety - ToolCrusher & SmartCrusher on tool-use edge cases
    crusher     - SmartCrusher only
    window      - RollingWindow only
    pipeline    - Full transform pipeline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Benchmark suite definitions
BENCHMARK_SUITES = {
    "all": [
        "benchmarks/bench_transforms.py",
        "benchmarks/bench_relevance.py",
        "benchmarks/bench_tool_use_safety.py",
    ],
    "latency": [],  # Standalone script: python benchmarks/bench_latency.py
    "quality": [],  # Standalone script: python benchmarks/bench_quality.py
    "transforms": [
        "benchmarks/bench_transforms.py",
    ],
    "relevance": [
        "benchmarks/bench_relevance.py",
    ],
    "tool-safety": [
        "benchmarks/bench_tool_use_safety.py",
    ],
    "crusher": [
        "benchmarks/bench_transforms.py::TestSmartCrusherBenchmarks",
    ],
    "aligner": [
        "benchmarks/bench_transforms.py::TestCacheAlignerBenchmarks",
    ],
    "window": [
        "benchmarks/bench_transforms.py::TestRollingWindowBenchmarks",
    ],
    "pipeline": [
        "benchmarks/bench_transforms.py::TestTransformPipelineBenchmarks",
    ],
    "bm25": [
        "benchmarks/bench_relevance.py::TestBM25Benchmarks",
    ],
    "hybrid": [
        "benchmarks/bench_relevance.py::TestHybridBenchmarks",
    ],
}

# Performance targets (mean time in microseconds)
PERFORMANCE_TARGETS = {
    # SmartCrusher – statistical analysis scales O(n); these reflect BM25 tier
    "test_compress_100_items": 500000,  # 500ms (statistical analysis on 100 items)
    "test_compress_1000_items": 5000000,  # 5s  (1000 items)
    "test_compress_10000_items": 60000000,  # 60s  (stress test)
    # CacheAligner – pure string ops, very fast
    "test_date_extraction": 1000,  # 1ms
    "test_hash_computation": 1000,  # 1ms
    # RollingWindow – O(n) scan
    "test_window_50_turns": 15000,  # 15ms
    "test_window_200_turns": 30000,  # 30ms
    "test_single_item": 100,  # 0.1ms
    "test_batch_100": 1000,  # 1ms
    "test_batch_1000": 10000,  # 10ms
    # Pipeline – dominated by SmartCrusher cost
    "test_pipeline_simple": 5000,  # 5ms (no large arrays)
    "test_pipeline_agentic": 30000000,  # 30s (50 turns × SmartCrusher)
    "test_pipeline_rag": 50000,  # 50ms
    # Tool-use safety benchmarks (ToolCrusher — simple fixed rules)
    "test_compress_log_100_items": 2000,  # 2ms
    "test_compress_log_1000_items": 15000,  # 15ms
    "test_compress_diagnostics_200": 5000,  # 5ms
    "test_compress_stack_traces_100": 3000,  # 3ms
    "test_compress_multi_turn_5_turns": 10000,  # 10ms
    "test_compress_multi_turn_20_turns": 40000,  # 40ms
    "test_deeply_nested_schema_5_levels": 5000,  # 5ms
    # Tool-use safety benchmarks (SmartCrusher — statistical)
    "test_compress_log_with_errors_100": 150000,  # 150ms
    "test_compress_log_with_errors_500": 750000,  # 750ms
    "test_compress_diagnostics_100": 150000,  # 150ms
    "test_compress_metrics_with_spike": 150000,  # 150ms
    "test_plain_text_passthrough_1000_chars": 5000,  # 5ms (fast passthrough)
}


def run_benchmarks(
    suite: str,
    output_json: str | None = None,
    compare: str | None = None,
    verbose: bool = False,
    extra_args: list[str] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Run benchmark suite via pytest.

    Args:
        suite: Name of benchmark suite to run.
        output_json: Path to save JSON results.
        compare: Path to baseline JSON for comparison.
        verbose: Enable verbose output.
        extra_args: Additional pytest arguments.

    Returns:
        Tuple of (exit_code, results_dict).
    """
    if suite not in BENCHMARK_SUITES:
        print(f"Error: Unknown suite '{suite}'")
        print(f"Available suites: {', '.join(BENCHMARK_SUITES.keys())}")
        return 1, None

    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--benchmark-only",
        "--benchmark-sort=name",
    ]

    # Add test files/patterns
    cmd.extend(BENCHMARK_SUITES[suite])

    # Add output options
    if output_json:
        cmd.extend(["--benchmark-json", output_json])

    # Add comparison
    if compare:
        cmd.extend(["--benchmark-compare", compare])

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add extra args
    if extra_args:
        cmd.extend(extra_args)

    # Run benchmarks
    print(f"Running {suite} benchmarks...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)

    # Load results if saved
    results = None
    if output_json and Path(output_json).exists():
        with open(output_json) as f:
            results = json.load(f)

    return result.returncode, results


def generate_markdown_report(
    results: dict[str, Any],
    output_path: str,
    include_targets: bool = True,
) -> None:
    """Generate markdown report from benchmark results.

    Args:
        results: Benchmark results dictionary (from pytest-benchmark JSON).
        output_path: Path to write markdown file.
        include_targets: Include performance target comparison.
    """
    lines = []

    # Header
    lines.append("# Headroom SDK Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Machine info
    if "machine_info" in results:
        info = results["machine_info"]
        lines.append("## Environment")
        lines.append("")
        lines.append(f"- **Machine**: {info.get('machine', 'unknown')}")
        lines.append(f"- **Processor**: {info.get('processor', 'unknown')}")
        lines.append(f"- **Python**: {info.get('python_version', 'unknown')}")
        lines.append("")

    # Summary table
    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Test | Mean | StdDev | Min | Max | Target | Status |")
    lines.append("|------|------|--------|-----|-----|--------|--------|")

    benchmarks = results.get("benchmarks", [])
    passed = 0
    failed = 0

    for bench in benchmarks:
        name = bench["name"]
        stats = bench["stats"]

        mean_us = stats["mean"] * 1_000_000  # Convert to microseconds
        stddev_us = stats["stddev"] * 1_000_000
        min_us = stats["min"] * 1_000_000
        max_us = stats["max"] * 1_000_000

        # Format times
        mean_str = _format_time(mean_us)
        stddev_str = _format_time(stddev_us)
        min_str = _format_time(min_us)
        max_str = _format_time(max_us)

        # Check target
        test_name = name.split("::")[-1]
        target = PERFORMANCE_TARGETS.get(test_name)

        if target:
            target_str = _format_time(target)
            if mean_us <= target:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
        else:
            target_str = "-"
            status = "-"

        lines.append(
            f"| `{test_name}` | {mean_str} | {stddev_str} | {min_str} | {max_str} | {target_str} | {status} |"
        )

    lines.append("")

    # Summary stats
    total = passed + failed
    if total > 0:
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Passed**: {passed}/{total} ({100 * passed / total:.0f}%)")
        lines.append(f"- **Failed**: {failed}/{total} ({100 * failed / total:.0f}%)")
        lines.append("")

    # Performance notes
    lines.append("## Performance Targets")
    lines.append("")
    lines.append("| Component | Target | Notes |")
    lines.append("|-----------|--------|-------|")
    lines.append("| SmartCrusher (100 items) | < 500ms | Statistical BM25 analysis |")
    lines.append("| SmartCrusher (1000 items) | < 5s | Statistical BM25 analysis |")
    lines.append("| SmartCrusher (10000 items) | < 60s | Stress test |")
    lines.append("| CacheAligner | < 1ms | Date extraction + hash |")
    lines.append("| RollingWindow (50 turns) | < 15ms | Long conversation |")
    lines.append("| RollingWindow (200 turns) | < 30ms | Stress test |")
    lines.append("| BM25Scorer (batch 100) | < 1ms | Zero dependencies |")
    lines.append("| HybridScorer (batch 100) | < 50ms | With embeddings |")
    lines.append("| ToolCrusher log 100 items | < 2ms | Fixed-rule compression |")
    lines.append("| ToolCrusher log 1000 items | < 15ms | Fixed-rule compression |")
    lines.append("| ToolCrusher 5-turn agent | < 10ms | Multi-step conversation |")
    lines.append("| ToolCrusher 20-turn agent | < 40ms | Long agentic session |")
    lines.append("| SmartCrusher log w/errors 100 | < 150ms | Error-preserving statistical |")
    lines.append("| SmartCrusher plain text 1000c | < 5ms | Fast passthrough path |")
    lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {output_path}")


def _format_time(microseconds: float) -> str:
    """Format time value with appropriate unit."""
    if microseconds < 1000:
        return f"{microseconds:.1f}us"
    elif microseconds < 1_000_000:
        return f"{microseconds / 1000:.2f}ms"
    else:
        return f"{microseconds / 1_000_000:.2f}s"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Headroom SDK benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--suite",
        "-s",
        choices=list(BENCHMARK_SUITES.keys()),
        default="all",
        help="Benchmark suite to run (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output markdown report path",
    )

    parser.add_argument(
        "--json",
        "-j",
        help="Save raw JSON results to path",
    )

    parser.add_argument(
        "--compare",
        "-c",
        help="Compare against baseline JSON",
    )

    parser.add_argument(
        "--save-baseline",
        help="Save results as baseline (alias for --json)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional pytest arguments",
    )

    args = parser.parse_args()

    # Handle save-baseline as alias
    json_output = args.json or args.save_baseline

    # Latency suite is a standalone script, not pytest-benchmark
    if args.suite == "latency":
        cmd = [sys.executable, "benchmarks/bench_latency.py"]
        if args.output:
            cmd.extend(["--output", args.output])
        if json_output:
            cmd.extend(["--json", json_output])
        if args.verbose:
            cmd.append("-v")
        print("Delegating to latency benchmark script...")
        return subprocess.run(cmd).returncode

    # Quality suite is a standalone script, not pytest-benchmark
    if args.suite == "quality":
        cmd = [sys.executable, "benchmarks/bench_quality.py"]
        if args.output:
            cmd.extend(["--output", args.output])
        if json_output:
            cmd.extend(["--json", json_output])
        if args.verbose:
            cmd.append("-v")
        print("Delegating to quality benchmark script...")
        return subprocess.run(cmd).returncode

    # Run benchmarks
    exit_code, results = run_benchmarks(
        suite=args.suite,
        output_json=json_output,
        compare=args.compare,
        verbose=args.verbose,
        extra_args=args.pytest_args,
    )

    # Generate markdown report if requested
    if args.output and results:
        generate_markdown_report(results, args.output)
    elif args.output and json_output:
        # Load results from saved JSON
        with open(json_output) as f:
            results = json.load(f)
        generate_markdown_report(results, args.output)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
