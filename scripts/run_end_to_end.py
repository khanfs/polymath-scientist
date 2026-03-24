"""
End-to-end Polymath Scientist pipeline runner.

Runs all four stages in sequence with clear progress reporting,
pre-flight checks, and recovery hints on failure.

Usage
-----
    # Quick smoke-test (tiny sample data, fast):
    python scripts/run_end_to_end.py --mode sample

    # Full pipeline (all data, slow — requires GPU/MPS):
    python scripts/run_end_to_end.py --mode full

Stages
------
    1. Data pipeline   — download, clean, balance, split
    2. Student training — fine-tune DistilGPT-2 on the splits
    3. Distillation    — multi-teacher representation distillation
    4. Evaluation      — structured prompt-based evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("end_to_end")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info("  %s", title)
    logger.info("=" * width)


def _check_splits_exist() -> bool:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if not splits_dir.exists():
        return False
    return bool(list(splits_dir.glob("*_split.json")))


def _check_model_exists() -> bool:
    model_dir = PROJECT_ROOT / "fine_tuned_model"
    if not model_dir.exists():
        return False
    has_config  = (model_dir / "config.json").exists()
    has_weights = (
        list(model_dir.glob("*.safetensors")) or
        list(model_dir.glob("pytorch_model*.bin"))
    )
    return has_config and bool(has_weights)


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    return f"{secs // 60}m {secs % 60}s"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Data pipeline
# ──────────────────────────────────────────────────────────────────────────────

def stage_data(sample_mode: bool) -> None:
    _banner("Stage 1 / 4 — Data Pipeline")
    t0 = time.time()

    from transformers import AutoTokenizer
    from src.data.multidisciplinary_data import MultidisciplinaryConfig, MultidisciplinaryDataPreparer
    from src.data.cross_validation import CrossValidationConfig, CrossValidator
    from src.data.validate_datasets import validate_split_directory

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if sample_mode:
        logger.info("Sample mode: loading a small slice of each dataset.")
        max_samples      = 200
        samples_per_source = 50
        min_samples_per_class = 3
    else:
        logger.info("Full mode: loading complete datasets (this will take a while).")
        max_samples      = 10_000
        samples_per_source = 5_000
        min_samples_per_class = 10

    preparer = MultidisciplinaryDataPreparer(
        tokenizer=tokenizer,
        config=MultidisciplinaryConfig(
            base_path=PROJECT_ROOT,
            use_cache=True,
            sample_mode=sample_mode,
            max_samples=max_samples,
            samples_per_source=samples_per_source,
            min_word_count=50,
            max_length=512,
            # In sample mode each topic has ~50 texts after filtering, so
            # the default min_samples_per_category=100 drops everything.
            # Use 10 for sample mode; keep 100 for full mode.
            min_samples_per_category=10 if sample_mode else 100,
            run_vocab_analysis=True,
            run_dataset_analysis=True,
        ),
    )

    texts, sources = preparer.prepare_multidisciplinary_data()
    logger.info("Dataset prepared: %d texts from %d unique sources",
                len(texts), len(set(sources)))

    if len(texts) < 10:
        raise RuntimeError(
            f"Only {len(texts)} texts loaded after filtering. "
            "Check your internet connection and that the HuggingFace datasets "
            "are accessible, then re-run."
        )

    split_creator = CrossValidator(
        config=CrossValidationConfig(
            base_path=PROJECT_ROOT,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            min_samples_per_class=min_samples_per_class,
            save_splits=True,
            run_analysis=True,
        )
    )

    train_data, val_data, test_data = split_creator.create_splits(
        texts=texts,
        sources=sources,
    )

    logger.info(
        "Splits created — train=%d  val=%d  test=%d",
        len(train_data["texts"]),
        len(val_data["texts"]),
        len(test_data["texts"]),
    )

    splits_dir = PROJECT_ROOT / "data" / "splits"
    if splits_dir.exists():
        results = validate_split_directory(splits_dir)
        logger.info("Validated %d split files — all OK", len(results))

    logger.info("Stage 1 complete in %s", _elapsed(t0))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Student model training
# ──────────────────────────────────────────────────────────────────────────────

def stage_training(sample_mode: bool) -> None:
    _banner("Stage 2 / 4 — Student Model Fine-Tuning")
    t0 = time.time()

    if not _check_splits_exist():
        raise RuntimeError(
            "No split files found. Run Stage 1 first:\n"
            "    python scripts/run_end_to_end.py --mode sample"
        )

    from src.training.training import StudentModelTrainer, TrainingConfig

    config = TrainingConfig(
        base_path=PROJECT_ROOT,
        model_name="distilgpt2",
        batch_size=4 if sample_mode else 8,
        accumulation_steps=2 if sample_mode else 4,
        learning_rate=2e-5,
        epochs=1 if sample_mode else 3,
        max_seq_length=512,
        warmup_steps=20 if sample_mode else 100,
        weight_decay=0.01,
        logging_steps=5 if sample_mode else 10,
        save_steps=100 if sample_mode else 500,
        eval_steps=100 if sample_mode else 500,
    )

    trainer = StudentModelTrainer(config=config)
    trainer.train()

    logger.info("Stage 2 complete in %s", _elapsed(t0))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Knowledge distillation
# ──────────────────────────────────────────────────────────────────────────────

def stage_distillation(sample_mode: bool, resume_epoch: int = 0) -> None:
    _banner("Stage 3 / 4 — Multi-Teacher Knowledge Distillation")
    t0 = time.time()

    if not _check_splits_exist():
        raise RuntimeError("No split files found — run Stage 1 first.")

    if not _check_model_exists():
        raise RuntimeError(
            "No fine-tuned model found in fine_tuned_model/.\n"
            "Run Stage 2 first:  python scripts/run_end_to_end.py --mode sample"
        )

    from src.training.distillation import DistillationConfig, PolymathDistillationTrainer

    config = DistillationConfig(
        base_path=PROJECT_ROOT,
        # max_length=256 during distillation (vs 512 in training) to reduce
        # memory pressure from three concurrent teacher forward passes on MPS.
        # This prevents kernel panics from memory exhaustion.
        max_length=128 if sample_mode else 256,
        batch_size=1,
        learning_rate=2e-5,
        epochs=1 if sample_mode else 3,
        alpha_distill=1.0,
        alpha_lm=1.0,
        warmup_steps=10 if sample_mode else 100,
        student_hidden_size=768,
        teacher_hidden_size=768,
        projection_dim=256,
        alpha_collapse=0.1,
        resume_from_epoch=resume_epoch,
    )

    trainer = PolymathDistillationTrainer(config=config)

    student_tokenizer, student_model = trainer.load_student_model()
    teachers = trainer.load_teacher_models()

    train_dataset = trainer.load_and_tokenize_split("train", student_tokenizer)

    # Load val split if available
    val_dataset = None
    val_path = PROJECT_ROOT / config.splits_dir / "val_split.json"
    if val_path.exists():
        val_dataset = trainer.load_and_tokenize_split("val", student_tokenizer)
        logger.info("Validation split loaded (%d examples)", len(val_dataset))
    else:
        logger.warning("No val split found — best-model tracking disabled.")

    trainer.train(
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        teachers=teachers,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Stage 3 complete in %s", _elapsed(t0))


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def stage_evaluation() -> None:
    _banner("Stage 4 / 4 — Structured Prompt Evaluation")
    t0 = time.time()

    if not _check_model_exists():
        raise RuntimeError(
            "No model found — run Stages 1–3 first."
        )

    from src.training.evaluate_polymath import EvaluationConfig, PolymathEvaluator

    evaluator = PolymathEvaluator(config=EvaluationConfig(base_path=PROJECT_ROOT))
    output_path = evaluator.evaluate_and_save()
    logger.info("Evaluation results saved to %s", output_path)
    logger.info("Stage 4 complete in %s", _elapsed(t0))

    # Print a few sample responses
    import json
    with output_path.open() as fh:
        results = json.load(fh)

    print()
    print("─" * 60)
    print("  Sample outputs")
    print("─" * 60)
    for item in results["results"][:3]:
        print(f"\n  [{item['category'].upper()}]")
        print(f"  Prompt:   {item['prompt']}")
        response = item["response"][len(item["prompt"]):].strip()
        preview   = response[:300] + ("..." if len(response) > 300 else "")
        print(f"  Response: {preview}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymath Scientist end-to-end pipeline")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="'sample' for a quick smoke-test, 'full' for the complete pipeline",
    )
    parser.add_argument(
        "--stage",
        choices=["data", "train", "distill", "eval", "all"],
        default="all",
        help="Run a specific stage only, or 'all' (default)",
    )
    parser.add_argument(
        "--resume-from-epoch",
        type=int,
        default=0,
        help="Resume distillation from checkpoint_epoch_N (0 = start from scratch)",
    )
    args = parser.parse_args()

    sample_mode = args.mode == "sample"
    stage       = args.stage

    logger.info(
        "Starting pipeline  mode=%s  stage=%s",
        args.mode, stage,
    )

    total_start = time.time()

    try:
        if stage in ("data",  "all"):
            stage_data(sample_mode)
        if stage in ("train", "all"):
            stage_training(sample_mode)
        if stage in ("distill", "all"):
            stage_distillation(sample_mode, resume_epoch=args.resume_from_epoch)
            # Always run evaluation after distillation completes, even when
            # --stage distill was specified explicitly.
            if stage == "distill":
                stage_evaluation()
        elif stage == "eval":
            stage_evaluation()

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        logger.debug("Full traceback:", exc_info=True)
        print()
        print("  Recovery hint:")
        print(f"  {exc}")
        print()
        sys.exit(1)

    logger.info("All stages complete in %s", _elapsed(total_start))


if __name__ == "__main__":
    main()
