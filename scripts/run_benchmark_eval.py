#!/usr/bin/env python3
"""
run_benchmark_eval.py — Runner script for Polymath Scientist benchmark evaluation.

Runs perplexity-based zero-shot multiple-choice evaluation on:
  - SciQ (science questions, 4-choice)
  - MMLU high_school_biology (4-choice)
  - MMLU high_school_chemistry (4-choice)
  - MMLU high_school_physics (4-choice)

Usage:
    python scripts/run_benchmark_eval.py
    python scripts/run_benchmark_eval.py --max-samples 200   # quick run
    python scripts/run_benchmark_eval.py --benchmarks sciq   # single benchmark
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.benchmark_eval import BenchmarkConfig, BenchmarkEvaluator


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+",
                        default=["sciq", "mmlu_bio", "mmlu_chem", "mmlu_phys"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        log_wandb=not args.no_wandb,
    )

    print(f"\nBenchmarks: {config.benchmarks}")
    print(f"Max samples per benchmark: {config.max_samples}")
    print(f"Device: {config.device}\n")

    evaluator = BenchmarkEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
