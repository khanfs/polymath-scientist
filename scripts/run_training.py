"""
Run student model fine-tuning.

This script:
- loads the train/validation splits
- loads the base DistilGPT2 student model
- fine-tunes the model on the multidisciplinary dataset
- saves the fine-tuned model
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.training.training import StudentModelTrainer, TrainingConfig


logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting student model fine-tuning run...")

    splits_dir = project_root / "data" / "splits"
    required_files = [
        splits_dir / "train_split.json",
        splits_dir / "val_split.json",
    ]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_str = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            f"Required split file(s) not found: {missing_str}. "
            "Run the data pipeline first."
        )

    config = TrainingConfig(
        base_path=project_root,
        splits_dir="data/splits",
        model_output_dir="fine_tuned_model",
        log_dir="logs",
        cache_dir="cache",
        model_name="distilgpt2",
        batch_size=8,
        accumulation_steps=4,
        learning_rate=2e-5,
        epochs=3,
        max_seq_length=512,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        use_mps=True,
    )

    trainer = StudentModelTrainer(config=config)
    trainer.train()

    logger.info("Student model fine-tuning completed successfully.")


if __name__ == "__main__":
    main()