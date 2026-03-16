"""
Prepare a multidisciplinary scientific dataset.

This module:
- loads cleaned texts from configured scientific sources
- filters very short texts
- balances topics across physics, chemistry, and biology
- optionally analyzes vocabulary coverage
- optionally analyzes and shuffles the final dataset

It is intended to be imported from scripts or notebooks rather than
executed directly.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.data.load_data import DataLoadingConfig, ScientificDataLoader
from src.helpers.data_caching import CacheConfig, ScientificDataCache
from src.helpers.shuffling_analysis import AnalysisConfig, ScientificDatasetAnalyzer
from src.helpers.topic_balancing import ScientificTopicBalancer, TopicBalancingConfig
from src.helpers.vocabulary_analysis import (
    ScientificVocabularyAnalyzer,
    VocabAnalysisConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class MultidisciplinaryConfig:
    """Configuration for multidisciplinary dataset preparation."""

    base_path: Path = PROJECT_ROOT
    cache_dir: str = "cache"
    log_dir: str = "logs"
    analysis_dir: str = "analysis"

    use_cache: bool = True
    sample_mode: bool = False
    max_samples: int = 10000
    samples_per_source: int = 5000
    min_word_count: int = 50
    min_samples_per_category: int = 100
    max_length: int = 512

    run_vocab_analysis: bool = True
    run_dataset_analysis: bool = True


class MultidisciplinaryDataPreparer:
    """Prepare a balanced multidisciplinary corpus."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[MultidisciplinaryConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or MultidisciplinaryConfig()
        self.logger = self._setup_logger()

        self.data_loader = ScientificDataLoader(
            DataLoadingConfig(
                base_path=self.config.base_path,
                cache_dir=self.config.cache_dir,
                log_dir=self.config.log_dir,
                use_cache=self.config.use_cache,
                sample_mode=self.config.sample_mode,
                max_samples=self.config.max_samples,
                samples_per_source=self.config.samples_per_source,
            )
        )

        self.topic_balancer = ScientificTopicBalancer(
            TopicBalancingConfig(
                max_samples_per_category=self.config.max_samples,
                min_samples_per_category=self.config.min_samples_per_category,
            )
        )

        self.vocab_analyzer = ScientificVocabularyAnalyzer(
            tokenizer=self.tokenizer,
            config=VocabAnalysisConfig(),
        )

        self.dataset_analyzer = ScientificDatasetAnalyzer(
            config=AnalysisConfig(
                output_dir=self.config.base_path / self.config.analysis_dir,
            ),
            tokenizer=self.tokenizer,
        )

        self.cache_manager = ScientificDataCache(
            CacheConfig(
                cache_dir=self.config.base_path / self.config.cache_dir,
                cache_version="1.0",
            )
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for dataset preparation."""
        log_dir = self.config.base_path / self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "multidisciplinary_data.log"

        logger = logging.getLogger("multidisciplinary_data")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        return logger

    def _filter_short_texts(
        self,
        texts: List[str],
        sources: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Remove texts below the minimum word-count threshold."""
        filtered_texts: List[str] = []
        filtered_sources: List[str] = []

        for text, source in zip(texts, sources):
            if len(text.split()) >= self.config.min_word_count:
                filtered_texts.append(text)
                filtered_sources.append(source)

        self.logger.info(
            "Filtered short texts: %s -> %s",
            len(texts),
            len(filtered_texts),
        )
        return filtered_texts, filtered_sources

    def _truncate_for_analysis(self, texts: List[str]) -> Tuple[List[str], List[List[int]]]:
        """Tokenize and truncate texts for downstream analysis."""
        truncated_texts: List[str] = []
        tokenized_texts: List[List[int]] = []

        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                max_length=self.config.max_length,
                truncation=True,
                add_special_tokens=True,
            )
            tokenized_texts.append(tokens)
            truncated_texts.append(
                self.tokenizer.decode(tokens, skip_special_tokens=True)
            )

        return truncated_texts, tokenized_texts

    def prepare_multidisciplinary_data(self) -> Tuple[List[str], List[str]]:
        """Prepare the full multidisciplinary dataset."""
        cache_fingerprint = {
            "sample_mode": self.config.sample_mode,
            "max_samples": self.config.max_samples,
            "samples_per_source": self.config.samples_per_source,
            "min_word_count": self.config.min_word_count,
            "min_samples_per_category": self.config.min_samples_per_category,
            "max_length": self.config.max_length,
        }

        if self.config.use_cache:
            cache_key = self.cache_manager.make_cache_key(
                base_name="multidisciplinary_dataset",
                config_fingerprint=cache_fingerprint,
            )
            cached = self.cache_manager.load(cache_key)
            if cached is not None:
                self.logger.info("Loaded multidisciplinary dataset from cache.")
                return cached

        self.logger.info("Loading raw scientific datasets...")
        texts, sources = self.data_loader.load_all_datasets()

        self.logger.info("Filtering short texts...")
        texts, sources = self._filter_short_texts(texts, sources)

        self.logger.info("Balancing topics...")
        texts, sources, balanced_topics = self.topic_balancer.balance_topics(texts, sources)

        self.logger.info("Preparing tokenized views for analysis...")
        truncated_texts, tokenized_texts = self._truncate_for_analysis(texts)

        if self.config.run_vocab_analysis:
            self.logger.info("Running vocabulary analysis...")
            try:
                self.vocab_analyzer.analyze_vocabulary_coverage(
                    tokenized_texts,
                    output_dir=self.config.base_path / self.config.analysis_dir,
                    sample_size=min(len(tokenized_texts), 10000),
                )
            except Exception as exc:
                self.logger.warning("Vocabulary analysis failed: %s", exc)

        final_texts = truncated_texts
        final_sources = sources

        if self.config.run_dataset_analysis:
            self.logger.info("Running dataset analysis and deterministic shuffling...")
            try:
                self.dataset_analyzer.analyze_dataset(final_texts, final_sources)
                final_texts, final_sources = self.dataset_analyzer.shuffle_data(
                    final_texts,
                    final_sources,
                )
            except Exception as exc:
                self.logger.warning("Dataset analysis failed: %s", exc)

        if self.config.use_cache:
            self.cache_manager.save(
                data=(final_texts, final_sources),
                base_name="multidisciplinary_dataset",
                config_fingerprint=cache_fingerprint,
                extra_metadata={
                    "num_samples": len(final_texts),
                    "source_distribution": dict(Counter(final_sources)),
                    "topic_distribution": dict(Counter(balanced_topics)),
                },
            )

        self.logger.info("Final dataset size: %s", len(final_texts))
        for source, count in Counter(final_sources).items():
            pct = (count / len(final_texts) * 100.0) if final_texts else 0.0
            self.logger.info("%s: %s texts (%.2f%%)", source, count, pct)

        return final_texts, final_sources