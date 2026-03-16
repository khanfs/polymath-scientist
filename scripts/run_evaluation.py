"""
Run structured evaluation for the Polymath student model.

This script:
- loads a saved model checkpoint
- runs structured prompt-based evaluation
- saves outputs and metadata to JSON
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.training.evaluate_polymath import EvaluationConfig, PolymathEvaluator


logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting polymath model evaluation run...")

    config = EvaluationConfig(
        base_path=project_root,
        model_dir="fine_tuned_model",
        output_dir="logs/evaluation",
        log_dir="logs",
    )

    evaluator = PolymathEvaluator(config=config)
    output_file = evaluator.evaluate_and_save()

    logger.info("Evaluation completed successfully.")
    logger.info("Results saved to: %s", output_file)


if __name__ == "__main__":
    main()