"""
Batch processing utilities for scientific text tokenization.

This module provides lightweight batching and tokenization helpers for dataset
preparation. It is intentionally narrow in scope and does not implement topic
classification or dataset balancing logic.

It is designed to be imported from the data pipeline rather than executed
directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from transformers import AutoTokenizer


LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for text batch processing."""

    model_name: str = "distilgpt2"
    max_length: int = 512
    batch_size: int = 32
    truncation: bool = True
    padding: str = "max_length"


class ScientificTextProcessor:
    """Utility for batching and tokenizing scientific text."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def batch_generator(self, texts: List[str]) -> Generator[List[str], None, None]:
        """Yield successive batches of texts."""
        for i in range(0, len(texts), self.config.batch_size):
            yield texts[i : i + self.config.batch_size]

    def tokenize_batch(
        self,
        texts: List[str],
        return_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """Tokenize a single batch of texts."""
        if not texts:
            return []

        encodings = self.tokenizer(
            texts,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            padding=self.config.padding,
            return_attention_mask=True,
        )

        results: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            record: Dict[str, Any] = {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
            }
            if return_text:
                record["text"] = text
            results.append(record)

        return results

    def process_dataset(
        self,
        texts: List[str],
        return_text: bool = False,
    ) -> List[Dict[str, Any]]:
        """Tokenize a full dataset in batches."""
        results: List[Dict[str, Any]] = []

        for batch in self.batch_generator(texts):
            results.extend(self.tokenize_batch(batch, return_text=return_text))

        LOGGER.info("Processed %s texts into tokenized records.", len(results))
        return results