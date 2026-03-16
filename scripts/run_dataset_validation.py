"""
Run dataset validation.

This script validates all dataset split files and writes
validation metadata alongside them.
"""

from pathlib import Path

from src.data.validate_datasets import validate_split_directory


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    splits_dir = project_root / "data" / "splits"

    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    print(f"Validating dataset splits in: {splits_dir}")

    results = validate_split_directory(splits_dir)

    print(f"\nValidated {len(results)} dataset file(s).")


if __name__ == "__main__":
    main()