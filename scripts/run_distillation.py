"""
Run knowledge distillation training.

This script:
- loads the fine-tuned student model
- loads the teacher models
- loads and tokenizes the training and validation splits
- runs multi-teacher representation-level knowledge distillation
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.training.distillation import DistillationConfig, PolymathDistillationTrainer


logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting polymath distillation training run...")

    config = DistillationConfig(
        base_path=project_root,
        splits_dir="data/splits",
        model_output_dir="fine_tuned_model",
        log_dir="logs",
        max_length=512,
        batch_size=1,
        learning_rate=2e-5,
        epochs=3,
        alpha_distill=1.0,
        alpha_lm=1.0,
        num_workers=0,
        warmup_steps=100,
        # Projection head dimensions — both DistilGPT-2 and BERT-style
        # teachers have hidden_size=768, but their representation spaces
        # are incompatible.  Projection heads map both sides into a shared
        # 256-dimensional space before the distillation loss is computed.
        student_hidden_size=768,
        teacher_hidden_size=768,
        projection_dim=256,
    )

    trainer = PolymathDistillationTrainer(config=config)

    student_tokenizer, student_model = trainer.load_student_model()
    teachers = trainer.load_teacher_models()

    train_dataset = trainer.load_and_tokenize_split(
        split_name="train",
        tokenizer=student_tokenizer,
    )

    # Load validation split for best-model tracking.
    # If the val split does not exist, pass val_dataset=None to train()
    # and best-checkpoint saving will be skipped.
    splits_path = project_root / config.splits_dir
    val_split_path = splits_path / "val_split.json"

    val_dataset = None
    if val_split_path.exists():
        val_dataset = trainer.load_and_tokenize_split(
            split_name="val",
            tokenizer=student_tokenizer,
        )
    else:
        logger.warning(
            "Validation split not found at %s — best-model tracking disabled.",
            val_split_path,
        )

    trainer.train(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teachers=teachers,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Distillation training completed successfully.")


if __name__ == "__main__":
    main()
