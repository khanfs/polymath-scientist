"""
Dataset validation utilities.

This module provides:
1. Schema validation:
   - required keys exist
   - expected types are correct
   - lengths match across aligned fields

2. Checksum validation:
   - compute SHA256 fingerprint of dataset files
   - save validation metadata for reproducibility
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetValidationConfig:
    """Configuration for dataset validation."""

    required_list_keys: tuple[str, ...] = ("texts", "sources")
    optional_list_keys: tuple[str, ...] = ("topics",)
    optional_dict_keys: tuple[str, ...] = ("metadata",)
    metadata_suffix: str = ".metadata.json"
    preview_samples: int = 3
    require_non_empty_strings: bool = True


@dataclass
class DatasetValidationResult:
    """Structured validation result."""

    file_path: str
    file_name: str
    is_valid: bool
    num_examples: int
    checksum_sha256: str
    validated_at_utc: str
    schema_version: str
    keys_present: List[str]
    optional_keys_present: List[str]


def load_dataset_file(dataset_path: str | Path) -> Dict[str, Any]:
    """Load a dataset JSON file."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_file_checksum(dataset_path: str | Path) -> str:
    """Compute SHA256 checksum for a file."""
    dataset_path = Path(dataset_path)
    sha256 = hashlib.sha256()

    with dataset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def _validate_string_list(
    values: List[Any],
    key: str,
    require_non_empty_strings: bool = True,
) -> None:
    """Validate that a list contains strings, optionally non-empty."""
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"All items in '{key}' must be strings")

    if require_non_empty_strings and not all(item.strip() for item in values):
        raise ValueError(f"All items in '{key}' must be non-empty strings")


def validate_dataset_schema(
    dataset: Dict[str, Any],
    config: Optional[DatasetValidationConfig] = None,
) -> None:
    """
    Validate dataset structure and alignment.

    Raises:
        ValueError: if schema validation fails.
        TypeError: if field types are invalid.
    """
    config = config or DatasetValidationConfig()

    # Required keys
    for key in config.required_list_keys:
        if key not in dataset:
            raise ValueError(f"Missing required key: '{key}'")

    # Required key types
    for key in config.required_list_keys:
        if not isinstance(dataset[key], list):
            raise TypeError(
                f"Expected '{key}' to be a list, got {type(dataset[key]).__name__}"
            )

    # Optional list key types
    for key in config.optional_list_keys:
        if key in dataset and not isinstance(dataset[key], list):
            raise TypeError(
                f"Expected optional key '{key}' to be a list, got {type(dataset[key]).__name__}"
            )

    # Optional dict key types
    for key in config.optional_dict_keys:
        if key in dataset and not isinstance(dataset[key], dict):
            raise TypeError(
                f"Expected optional key '{key}' to be a dict, got {type(dataset[key]).__name__}"
            )

    # Length alignment for list fields
    required_lengths = [len(dataset[key]) for key in config.required_list_keys]
    if len(set(required_lengths)) != 1:
        raise ValueError(
            "Length mismatch across required fields: "
            + ", ".join(f"{key}={len(dataset[key])}" for key in config.required_list_keys)
        )

    for key in config.optional_list_keys:
        if key in dataset and len(dataset[key]) != required_lengths[0]:
            raise ValueError(
                f"Length mismatch: '{key}' has length {len(dataset[key])}, "
                f"expected {required_lengths[0]}"
            )

    # Element-level validation
    for key in config.required_list_keys:
        _validate_string_list(
            dataset[key],
            key=key,
            require_non_empty_strings=config.require_non_empty_strings,
        )

    for key in config.optional_list_keys:
        if key in dataset:
            _validate_string_list(
                dataset[key],
                key=key,
                require_non_empty_strings=config.require_non_empty_strings,
            )

    # Basic sanity check
    if len(dataset["texts"]) == 0:
        raise ValueError("Dataset contains zero examples")


def summarize_dataset(
    dataset: Dict[str, Any],
    num_samples: int = 3,
) -> None:
    """Print a simple human-readable dataset summary."""
    total_examples = len(dataset["texts"])
    print(f"Loaded dataset with {total_examples} examples.")
    print(f"Keys present: {list(dataset.keys())}")

    if "sources" in dataset:
        unique_sources = sorted(set(dataset["sources"]))
        print(f"Unique sources ({len(unique_sources)}): {unique_sources}")

    if "topics" in dataset:
        unique_topics = sorted(set(dataset["topics"]))
        print(f"Unique topics ({len(unique_topics)}): {unique_topics}")

    if "metadata" in dataset:
        print(f"Metadata keys: {list(dataset['metadata'].keys())}")

    for i in range(min(num_samples, total_examples)):
        print(f"\nSample {i + 1}:")
        text_preview = dataset["texts"][i][:300]
        if len(dataset["texts"][i]) > 300:
            text_preview += "..."
        print(text_preview)
        print(f"Source: {dataset['sources'][i]}")

        if "topics" in dataset:
            print(f"Topic: {dataset['topics'][i]}")


def build_validation_result(
    dataset_path: str | Path,
    dataset: Dict[str, Any],
    checksum_sha256: str,
    config: Optional[DatasetValidationConfig] = None,
) -> DatasetValidationResult:
    """Build a structured validation result."""
    config = config or DatasetValidationConfig()
    dataset_path = Path(dataset_path)

    keys_present = [key for key in config.required_list_keys if key in dataset]
    optional_keys_present = [
        key for key in (*config.optional_list_keys, *config.optional_dict_keys)
        if key in dataset
    ]

    return DatasetValidationResult(
        file_path=str(dataset_path.resolve()),
        file_name=dataset_path.name,
        is_valid=True,
        num_examples=len(dataset["texts"]),
        checksum_sha256=checksum_sha256,
        validated_at_utc=datetime.now(timezone.utc).isoformat(),
        schema_version="1.1",
        keys_present=keys_present,
        optional_keys_present=optional_keys_present,
    )


def save_validation_metadata(
    result: DatasetValidationResult,
    dataset_path: str | Path,
    config: Optional[DatasetValidationConfig] = None,
) -> Path:
    """Save validation metadata alongside the dataset file."""
    config = config or DatasetValidationConfig()
    dataset_path = Path(dataset_path)

    metadata_path = dataset_path.with_name(dataset_path.name + config.metadata_suffix)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2, ensure_ascii=False)

    return metadata_path


def validate_dataset_file(
    dataset_path: str | Path,
    config: Optional[DatasetValidationConfig] = None,
    save_metadata: bool = True,
    print_summary: bool = True,
) -> DatasetValidationResult:
    """
    Load, validate, checksum, and optionally summarize a dataset file.
    """
    config = config or DatasetValidationConfig()

    dataset = load_dataset_file(dataset_path)
    validate_dataset_schema(dataset, config=config)
    checksum_sha256 = compute_file_checksum(dataset_path)

    result = build_validation_result(
        dataset_path=dataset_path,
        dataset=dataset,
        checksum_sha256=checksum_sha256,
        config=config,
    )

    if save_metadata:
        metadata_path = save_validation_metadata(result, dataset_path, config=config)
        print(f"Saved validation metadata to: {metadata_path}")

    if print_summary:
        summarize_dataset(dataset, num_samples=config.preview_samples)
        print(f"\nSHA256: {result.checksum_sha256}")

    return result


def compare_dataset_checksum(
    dataset_path: str | Path,
    expected_checksum: str,
) -> bool:
    """Compare the current file checksum against an expected checksum."""
    current_checksum = compute_file_checksum(dataset_path)
    return current_checksum == expected_checksum


def validate_split_directory(
    splits_dir: str | Path,
    config: Optional[DatasetValidationConfig] = None,
) -> List[DatasetValidationResult]:
    """Validate all *_split.json files in a directory."""
    splits_dir = Path(splits_dir)
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    results: List[DatasetValidationResult] = []
    for dataset_path in sorted(splits_dir.glob("*_split.json")):
        result = validate_dataset_file(
            dataset_path,
            config=config,
            save_metadata=True,
            print_summary=False,
        )
        results.append(result)

    return results