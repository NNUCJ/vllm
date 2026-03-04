# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helper to run the multimodal benchmark against multiple vLLM instances.

Example:
    python benchmarks/run_multimodal_benchmark_suite.py \\
        --ports 8001 8002 8003 8004 \\
        --concurrency 3 6 \\
        --log-dir /tmp/vllm_bench_logs
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run benchmark_mutimodal.py against multiple ports and "
                     "capture logs for each (supports multiple concurrency "
                     "values)."))
    parser.add_argument(
        "--benchmark-script",
        default="/home/intellif/chengjie/tyvllm_project/ty_vllm/tyvllm/"
        "benchmarks/benchmark_mutimodal.py",
        help="Path to benchmark_mutimodal.py.")
    parser.add_argument("--ports",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Ports for target vLLM instances.")
    parser.add_argument("--concurrency",
                        type=int,
                        nargs="+",
                        default=[3],
                        help="List of max concurrency values to test.")
    parser.add_argument("--model",
                        default="/data3/model/qwen3vl-8b/AWQ/AOT/4die",
                        help="Model path.")
    parser.add_argument("--tokenizer",
                        default="/data3/model/qwen3vl-8b/AWQ",
                        help="Tokenizer path.")
    parser.add_argument("--host",
                        default="0.0.0.0",
                        help="Host of the vLLM service.")
    parser.add_argument("--dataset-name",
                        default="random",
                        help="Benchmark dataset.")
    parser.add_argument("--random-input-len",
                        type=int,
                        default=128,
                        help="Random input length.")
    parser.add_argument("--random-output-len",
                        type=int,
                        default=256,
                        help="Random output length.")
    parser.add_argument("--request-rate",
                        type=int,
                        default=64,
                        help="Request rate for the benchmark.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=64,
                        help="Number of prompts to run.")
    parser.add_argument("--log-dir",
                        type=Path,
                        default=Path("benchmark_logs"),
                        help="Directory to write log files.")
    parser.add_argument(
        "--extra-args",
        default="",
        help=("Extra args appended to the benchmark command. "
              "Use quotes, e.g., \"--some-flag value\"."))
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print commands without executing them.")
    return parser.parse_args()


def build_command(args: argparse.Namespace, port: int,
                  concurrency: int) -> List[str]:
    cmd = [
        sys.executable,
        args.benchmark_script,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--host",
        args.host,
        "--dataset-name",
        args.dataset_name,
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        str(args.request_rate),
        "--num-prompts",
        str(args.num_prompts),
        "--port",
        str(port),
    ]
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def run_command(cmd: Iterable[str], log_path: Path,
                dry_run: bool) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY RUN] {' '.join(shlex.quote(part) for part in cmd)}")
        print(f"[DRY RUN] Log -> {log_path}")
        return subprocess.CompletedProcess(cmd, returncode=0)

    print(f"Running: {' '.join(shlex.quote(part) for part in cmd)}")
    print(f"Logging to: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(cmd,
                              stdout=log_file,
                              stderr=subprocess.STDOUT,
                              check=False)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for port in args.ports:
        for concurrency in args.concurrency:
            log_name = f"port{port}_mc{concurrency}_{timestamp}.log"
            log_path = args.log_dir / log_name
            cmd = build_command(args, port, concurrency)
            result = run_command(cmd, log_path, args.dry_run)
            results.append(((port, concurrency), result.returncode))
            if result.returncode != 0:
                print(
                    f"[WARN] Benchmark failed for port {port} with concurrency "
                    f"{concurrency} (exit code {result.returncode}).")

    failed = [item for item in results if item[1] != 0]
    if failed:
        print("\nSome benchmarks failed:")
        for (port, concurrency), code in failed:
            print(f"  port={port}, max-concurrency={concurrency}, exit={code}")
        return 1

    print("\nAll benchmarks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
