"""
Optional data augmentation utilities for scientific text.

This module is intentionally conservative. For scientific corpora, aggressive
augmentation strategies such as random synonym replacement, insertion, swapping,
or deletion can corrupt meaning. The utilities here therefore focus on
lightweight, lower-risk transformations and reproducible behaviour.

This module is designed to be imported from the data pipeline rather than
executed directly.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for conservative scientific text augmentation."""

    enabled: bool = False
    random_seed: int = 42
    preserve_original: bool = True
    lowercase_probability: float = 0.0
    whitespace_normalization: bool = True
    deduplicate_outputs: bool = True


class ScientificDataAugmenter:
    """
    Conservative augmenter for scientific text.

    Notes
    -----
    This class deliberately avoids aggressive augmentation methods that can
    alter scientific meaning. It is intended as an optional utility, not a
    required part of the pipeline.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        self.config = config or AugmentationConfig()
        self.rng = random.Random(self.config.random_seed)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse repeated whitespace while preserving text content."""
        return re.sub(r"\s+", " ", text).strip()

    def maybe_lowercase(self, text: str) -> str:
        """
        Optionally lowercase text.

        Disabled by default because lowercasing may damage scientific notation,
        abbreviations, gene/protein names, and domain-specific casing.
        """
        if self.rng.random() < self.config.lowercase_probability:
            return text.lower()
        return text

    def augment_text(self, text: str) -> List[str]:
        """Return one or more conservative variants of a text."""
        if not isinstance(text, str):
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        outputs: List[str] = []

        if self.config.preserve_original:
            outputs.append(cleaned)

        candidate = cleaned

        if self.config.whitespace_normalization:
            candidate = self.normalize_whitespace(candidate)

        candidate = self.maybe_lowercase(candidate)

        outputs.append(candidate)

        if self.config.deduplicate_outputs:
            deduped = []
            seen = set()
            for item in outputs:
                if item not in seen:
                    deduped.append(item)
                    seen.add(item)
            outputs = deduped

        return outputs

    def augment_dataset(self, texts: Iterable[str]) -> List[str]:
        """Augment an iterable of texts conservatively and reproducibly."""
        if not self.config.enabled:
            LOGGER.info("Data augmentation disabled; returning original texts only.")
            return [text for text in texts if isinstance(text, str) and text.strip()]

        augmented: List[str] = []
        for text in texts:
            augmented.extend(self.augment_text(text))

        if self.config.deduplicate_outputs:
            deduped = []
            seen = set()
            for item in augmented:
                if item not in seen:
                    deduped.append(item)
                    seen.add(item)
            augmented = deduped

        LOGGER.info("Augmented dataset size: %s", len(augmented))
        return augmented