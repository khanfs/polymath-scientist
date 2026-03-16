"""
Dataset splitting and topic classification utilities.

This module provides:
- stratified train / validation / test splitting
- cross-validation fold creation
- heuristic topic classification support
- dataset distribution logging

It is designed to be imported from scripts or notebooks rather than
executed directly.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.helpers.shuffling_analysis import AnalysisConfig, ScientificDatasetAnalyzer
from src.helpers.topic_balancing import ScientificTopicBalancer, TopicBalancingConfig


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CrossValidationConfig:
    """Configuration for dataset splitting and cross-validation."""

    base_path: Path = PROJECT_ROOT
    log_dir: str = "logs"
    splits_dir: str = "data/splits"
    n_splits: int = 5
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    min_samples_per_class: int = 10
    save_splits: bool = True
    batch_size: int = 1000
    run_analysis: bool = True


class CrossValidator:
    """Create stratified dataset splits and cross-validation folds."""

    def __init__(self, config: Optional[CrossValidationConfig] = None) -> None:
        self.config = config or CrossValidationConfig()
        self.logger = self._setup_logger()

        self.topic_balancer = ScientificTopicBalancer(
            TopicBalancingConfig(random_seed=self.config.random_state)
        )

        self.dataset_analyzer = ScientificDatasetAnalyzer(
            config=AnalysisConfig(
                seed=self.config.random_state,
                output_dir=self.config.base_path / "analysis",
            )
        )

    def _setup_logger(self) -> logging.Logger:
        """Configure logger for cross-validation workflow."""
        log_dir = self.config.base_path / self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "cross_validation.log"

        logger = logging.getLogger("cross_validator")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        return logger

    def create_splits(
        self,
        texts: List[str],
        sources: List[str],
        topics: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
        """
        Create stratified train / validation / test splits.

        Returns:
            (train_data, val_data, test_data)
        """
        self._validate_inputs(texts, sources, topics)

        if topics is None:
            self.logger.info("Classifying topics heuristically...")
            topics = self._classify_topics(texts)

        if self.config.run_analysis:
            self.logger.info("Analyzing full dataset distribution...")
            try:
                self.dataset_analyzer.analyze_dataset(texts, sources)
            except Exception as exc:
                self.logger.warning("Dataset analysis failed: %s", exc)

        train_indices, test_indices = self._create_stratified_split(
            sources=sources,
            topics=topics,
            test_size=self.config.test_size,
        )

        train_texts = [texts[i] for i in train_indices]
        train_sources = [sources[i] for i in train_indices]
        train_topics = [topics[i] for i in train_indices]

        final_train_indices_rel, val_indices_rel = self._create_stratified_split(
            sources=train_sources,
            topics=train_topics,
            test_size=self.config.val_size,
        )

        final_train_indices = [train_indices[i] for i in final_train_indices_rel]
        val_indices = [train_indices[i] for i in val_indices_rel]

        splits = {
            "train": self._gather_split_data(texts, sources, topics, final_train_indices),
            "val": self._gather_split_data(texts, sources, topics, val_indices),
            "test": self._gather_split_data(texts, sources, topics, test_indices),
        }

        if self.config.save_splits:
            self._save_splits(splits)

        self._log_split_statistics(splits)

        return splits["train"], splits["val"], splits["test"]

    def create_cv_folds(
        self,
        texts: List[str],
        sources: List[str],
        topics: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, List], Dict[str, List]]]:
        """
        Create stratified cross-validation folds.

        Returns:
            List of (train_data, val_data) tuples.
        """
        self._validate_inputs(texts, sources, topics)

        if topics is None:
            self.logger.info("Classifying topics heuristically for CV folds...")
            topics = self._classify_topics(texts)

        composite_labels = [f"{topic}_{source}" for topic, source in zip(topics, sources)]

        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        cv_folds: List[Tuple[Dict[str, List], Dict[str, List]]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(texts, composite_labels), start=1
        ):
            self.logger.info("Creating fold %s/%s", fold_idx, self.config.n_splits)

            train_data = self._gather_split_data(texts, sources, topics, train_idx.tolist())
            val_data = self._gather_split_data(texts, sources, topics, val_idx.tolist())

            cv_folds.append((train_data, val_data))

        self._log_cv_statistics(cv_folds)
        return cv_folds

    def _validate_inputs(
        self,
        texts: List[str],
        sources: List[str],
        topics: Optional[List[str]] = None,
    ) -> None:
        """Validate basic dataset inputs."""
        if len(texts) != len(sources):
            raise ValueError("Length mismatch between texts and sources")

        if not texts:
            raise ValueError("Input texts must not be empty")

        if topics is not None and len(topics) != len(texts):
            raise ValueError("Length mismatch between texts and topics")

    def _classify_topics(self, texts: List[str]) -> List[str]:
        """Classify topics heuristically using the topic balancer."""
        topics: List[str] = []

        for text in texts:
            topic, confidence, _ = self.topic_balancer.classify_topic(text)
            topics.append(topic if confidence > 0.0 else "other")

        return topics

    def _create_stratified_split(
        self,
        sources: List[str],
        topics: List[str],
        test_size: float,
    ) -> Tuple[List[int], List[int]]:
        """Create a stratified split based on composite topic-source labels."""
        composite_labels = [f"{topic}_{source}" for topic, source in zip(topics, sources)]

        label_counts = Counter(composite_labels)
        valid_label_set = {
            label
            for label, count in label_counts.items()
            if count >= self.config.min_samples_per_class
        }

        if not valid_label_set:
            raise ValueError(
                f"No classes meet minimum sample requirement of "
                f"{self.config.min_samples_per_class}"
            )

        valid_indices = [
            idx for idx, label in enumerate(composite_labels)
            if label in valid_label_set
        ]

        valid_labels = [composite_labels[i] for i in valid_indices]

        # Stratified splitting requires test_size >= n_classes.
        # With small datasets (e.g. sample mode) this can fail, so fall back
        # to simple random splitting when stratification is not feasible.
        n_classes = len(set(valid_labels))
        n_test = max(1, int(len(valid_indices) * test_size))
        use_stratify = n_test >= n_classes

        if not use_stratify:
            LOGGER.warning(
                "Stratified split not feasible: test_size=%d < n_classes=%d. "
                "Falling back to random split.",
                n_test,
                n_classes,
            )

        train_idx_rel, test_idx_rel = train_test_split(
            range(len(valid_indices)),
            test_size=test_size,
            stratify=valid_labels if use_stratify else None,
            random_state=self.config.random_state,
        )

        train_indices = [valid_indices[i] for i in train_idx_rel]
        test_indices = [valid_indices[i] for i in test_idx_rel]

        return train_indices, test_indices

    def _gather_split_data(
        self,
        texts: List[str],
        sources: List[str],
        topics: List[str],
        indices: List[int],
    ) -> Dict[str, List]:
        """Gather split data using the provided indices."""
        return {
            "texts": [texts[idx] for idx in indices],
            "sources": [sources[idx] for idx in indices],
            "topics": [topics[idx] for idx in indices],
        }

    def _save_splits(self, splits: Dict[str, Dict[str, List]]) -> None:
        """Save split data and metadata to disk."""
        splits_dir = self.config.base_path / self.config.splits_dir
        splits_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            split_path = splits_dir / f"{split_name}_split.json"

            serializable_data = {
                "texts": split_data["texts"],
                "sources": split_data["sources"],
                "topics": split_data["topics"],
                "metadata": {
                    "num_examples": len(split_data["texts"]),
                    "source_distribution": dict(Counter(split_data["sources"])),
                    "topic_distribution": dict(Counter(split_data["topics"])),
                    "split_name": split_name,
                    "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "random_state": self.config.random_state,
                },
            }

            with split_path.open("w", encoding="utf-8") as handle:
                json.dump(serializable_data, handle, indent=2, ensure_ascii=False)

            self.logger.info("Saved %s split to %s", split_name, split_path)

    def _log_split_statistics(self, splits: Dict[str, Dict[str, List]]) -> None:
        """Log statistics for train/val/test splits."""
        self.logger.info("Split Statistics:")

        for split_name, split_data in splits.items():
            self.logger.info("%s split:", split_name.upper())
            self.logger.info("Total examples: %s", len(split_data["texts"]))

            source_dist = Counter(split_data["sources"])
            self.logger.info("Source distribution:")
            for source, count in source_dist.most_common():
                percentage = (count / len(split_data["sources"])) * 100
                self.logger.info("  %s: %s (%.2f%%)", source, count, percentage)

            topic_dist = Counter(split_data["topics"])
            self.logger.info("Topic distribution:")
            for topic, count in topic_dist.most_common():
                percentage = (count / len(split_data["topics"])) * 100
                self.logger.info("  %s: %s (%.2f%%)", topic, count, percentage)

            text_lengths = [len(text.split()) for text in split_data["texts"]]
            if text_lengths:
                self.logger.info("Length statistics:")
                self.logger.info("  Average length: %.2f words", float(np.mean(text_lengths)))
                self.logger.info("  Median length: %.2f words", float(np.median(text_lengths)))
                self.logger.info("  Min length: %s words", min(text_lengths))
                self.logger.info("  Max length: %s words", max(text_lengths))

            self.logger.info("-" * 50)

    def _log_cv_statistics(
        self,
        cv_folds: List[Tuple[Dict[str, List], Dict[str, List]]],
    ) -> None:
        """Log statistics for cross-validation folds."""
        self.logger.info("Cross-Validation Statistics:")

        for fold_idx, (train_data, val_data) in enumerate(cv_folds, start=1):
            self.logger.info("Fold %s:", fold_idx)

            for split_name, split_data in [("Train", train_data), ("Val", val_data)]:
                self.logger.info("%s set statistics:", split_name)
                self.logger.info("Total examples: %s", len(split_data["texts"]))

                source_dist = Counter(split_data["sources"])
                topic_dist = Counter(split_data["topics"])

                self.logger.info("Source distribution:")
                for source, count in source_dist.most_common():
                    percentage = (count / len(split_data["texts"])) * 100
                    self.logger.info("  %s: %s (%.2f%%)", source, count, percentage)

                self.logger.info("Topic distribution:")
                for topic, count in topic_dist.most_common():
                    percentage = (count / len(split_data["texts"])) * 100
                    self.logger.info("  %s: %s (%.2f%%)", topic, count, percentage)