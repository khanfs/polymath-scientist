"""
Run quantitative evaluation of the distilled Polymath model.

Computes three metrics on the held-out test split:

  1. Perplexity per domain — baseline DistilGPT-2 vs distilled Polymath
  2. Representation alignment — cosine similarity to each teacher
  3. Domain specificity — does the student align most with the correct teacher?

Usage
-----
    python scripts/run_quantitative_eval.py

Results are saved to logs/evaluation/quantitative_results_<timestamp>.json
and printed as a formatted table.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.training.quantitative_eval import QuantEvalConfig, QuantitativeEvaluator


def main() -> None:
    config = QuantEvalConfig(
        base_path=PROJECT_ROOT,
        distilled_model_dir="fine_tuned_model",
        baseline_model_name="distilgpt2",
        splits_dir="data/splits",
        output_dir="logs/evaluation",
        max_length=512,
        batch_size=4,
    )

    evaluator = QuantitativeEvaluator(config=config)
    output_path = evaluator.evaluate()
    print(f"Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
