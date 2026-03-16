"""
Run the data preparation pipeline.

This script:
- loads raw scientific datasets
- prepares the multidisciplinary corpus
- creates train / validation / test splits
- validates saved split files
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from src.data.cross_validation import CrossValidationConfig, CrossValidator
from src.data.multidisciplinary_data import (
    MultidisciplinaryConfig,
    MultidisciplinaryDataPreparer,
)
from src.data.validate_datasets import validate_split_directory


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Prepare multidisciplinary dataset
    preparer_config = MultidisciplinaryConfig(
        base_path=project_root,
        cache_dir="cache",
        log_dir="logs",
        analysis_dir="analysis",
        use_cache=True,
        sample_mode=True,          # set to False for full pipeline
        max_samples=10,
        samples_per_source=5,
        min_word_count=50,
        max_length=512,
        run_vocab_analysis=True,
        run_dataset_analysis=True,
    )

    preparer = MultidisciplinaryDataPreparer(
        tokenizer=tokenizer,
        config=preparer_config,
    )

    texts, sources = preparer.prepare_multidisciplinary_data()

    # Step 2: Create train / val / test splits
    split_config = CrossValidationConfig(
        base_path=project_root,
        log_dir="logs",
        splits_dir="data/splits",
        n_splits=5,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        min_samples_per_class=3,   # lower for sample mode
        save_splits=True,
        batch_size=1000,
        run_analysis=True,
    )

    split_creator = CrossValidator(config=split_config)
    train_data, val_data, test_data = split_creator.create_splits(
        texts=texts,
        sources=sources,
        topics=None,
    )

    print("\nPipeline completed successfully.")
    print(f"Train examples: {len(train_data['texts'])}")
    print(f"Validation examples: {len(val_data['texts'])}")
    print(f"Test examples: {len(test_data['texts'])}")

    # Step 3: Validate saved split files
    splits_dir = project_root / "data" / "splits"
    if splits_dir.exists():
        print("\nValidating saved split files...")
        results = validate_split_directory(splits_dir)
        print(f"Validated {len(results)} split file(s).")


if __name__ == "__main__":
    main()