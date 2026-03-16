"""
Data loading utilities.

This module provides a reusable pipeline for loading and lightly filtering:
- scientific lay summaries
- Wikipedia science articles
- SciQ
- arXiv metadata snapshots

It is designed to return cleaned text samples and source labels for downstream
multidisciplinary corpus construction.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from datasets import load_dataset

from src.helpers.data_caching import CacheConfig, ScientificDataCache
from src.helpers.text_cleaning import ScientificTextCleaner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


@dataclass
class DataLoadingConfig:
    """Configuration for dataset loading and preprocessing."""

    base_path: Path = PROJECT_ROOT
    cache_dir: str = "cache"
    log_dir: str = "logs"
    batch_size: int = 32
    use_cache: bool = True

    sample_mode: bool = True
    max_samples: int = 10
    samples_per_source: int = 5

    arxiv_filename: str = "arxiv-metadata-oai-snapshot.json"


class ScientificDataLoader:
    """Load and preprocess scientific datasets."""

    def __init__(self, config: Optional[DataLoadingConfig] = None) -> None:
        self.config = config or DataLoadingConfig()
        self.logger = self._setup_logger()
        self.cleaner = ScientificTextCleaner()

        self.cache_manager = ScientificDataCache(
            CacheConfig(
                cache_dir=self.config.base_path / self.config.cache_dir,
                cache_version="1.0",
            )
        )

        self.wikipedia_keywords = {
            "physics", "chemistry", "biology", "quantum", "relativity",
            "molecule", "reaction", "atom", "cell", "gene", "protein",
            "dna", "rna", "enzyme", "thermodynamics", "electromagnetism",
        }

    def _setup_logger(self) -> logging.Logger:
        """Create a module-specific logger."""
        log_dir = self.config.base_path / self.config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "data_loading.log"

        logger = logging.getLogger("scientific_data_loader")
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

    def _batch_generator(self, dataset) -> Generator[List[Dict], None, None]:
        """Yield batches from a streaming dataset."""
        batch: List[Dict] = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _is_core_science_article(self, title: str, text: str) -> bool:
        """Lightweight heuristic filter for science-heavy Wikipedia pages."""
        sample = f"{title} {text[:500]}".lower()
        matches = sum(1 for keyword in self.wikipedia_keywords if keyword in sample)
        return matches >= 2

    def _is_relevant_arxiv_category(self, categories: str) -> bool:
        """Check whether an arXiv record belongs to relevant science categories."""
        relevant_categories = {
            "physics", "quant-ph", "cond-mat", "hep-th", "astro-ph",
            "chem-ph", "physics.chem-ph",
            "q-bio", "physics.bio-ph", "q-bio.GN", "q-bio.MN",
        }
        categories_set = set(categories.split())
        return bool(categories_set.intersection(relevant_categories))

    def load_lay_summaries(self) -> Tuple[List[str], List[str]]:
        """Load and clean biomedical abstracts from PubMed.

        Replaces tomasg25/scientific_lay_summarisation, which relied on a
        HuggingFace loading script no longer supported in datasets >= 3.0.
        ccdv/pubmed-summarization is a standard Parquet dataset requiring no
        loading script.  Abstracts are used as the biology-domain text source.
        """
        texts: List[str] = []
        sources: List[str] = []
        samples_loaded = 0

        self.logger.info("Loading PubMed abstracts (ccdv/pubmed-summarization)")

        dataset = load_dataset(
            "ccdv/pubmed-summarization",
            "document",
            split="train",
            streaming=True,
        )

        for batch in self._batch_generator(dataset):
            for sample in batch:
                text = sample.get("abstract", "") or sample.get("article", "")
                cleaned = self.cleaner.clean_scientific_text(
                    text,
                    source_type="lay_summary",
                )
                if cleaned:
                    texts.append(cleaned)
                    sources.append("pubmed")
                    samples_loaded += 1

                    if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                        break

            if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                break

        return texts, sources

    def load_wikipedia(self) -> Tuple[List[str], List[str]]:
        """Load and clean Wikipedia science articles.

        Uses wikimedia/wikipedia (20231101.en) which is a standard Parquet
        dataset.  The legacy wikipedia/20220301.en relied on a loading script
        no longer supported in datasets >= 3.0.
        """
        texts: List[str] = []
        sources: List[str] = []
        samples_loaded = 0

        self.logger.info("Loading Wikipedia dataset")

        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        for batch in self._batch_generator(dataset):
            for sample in batch:
                if not self._is_core_science_article(sample.get("title", ""), sample.get("text", "")):
                    continue

                cleaned = self.cleaner.clean_scientific_text(
                    sample.get("text", ""),
                    source_type="wikipedia",
                )
                if cleaned:
                    texts.append(cleaned)
                    sources.append("wikipedia")
                    samples_loaded += 1

                    # Use samples_per_source (consistent with other loaders).
                    # Previously used max_samples here, which could load
                    # up to max_samples/samples_per_source times more Wikipedia
                    # data than other sources in sample mode.
                    if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                        break

            if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                break

        return texts, sources

    def load_sciq(self) -> Tuple[List[str], List[str]]:
        """Load and clean the SciQ dataset."""
        texts: List[str] = []
        sources: List[str] = []
        samples_loaded = 0

        self.logger.info("Loading SciQ dataset")

        dataset = load_dataset(
            "sciq",
            split="train",
            streaming=True,
        )

        for batch in self._batch_generator(dataset):
            for sample in batch:
                combined = f"{sample.get('question', '')}\n{sample.get('support', '')}"
                cleaned = self.cleaner.clean_scientific_text(combined, source_type="sciq")
                if cleaned:
                    texts.append(cleaned)
                    sources.append("sciq")
                    samples_loaded += 1

                    if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                        break

            if self.config.sample_mode and samples_loaded >= self.config.samples_per_source:
                break

        return texts, sources

    def load_arxiv(self, file_path: str | Path) -> Tuple[List[str], List[str]]:
        """Load and clean an arXiv metadata snapshot stored as JSONL."""
        texts: List[str] = []
        sources: List[str] = []
        samples_loaded = 0
        file_path = Path(file_path)

        self.logger.info("Loading arXiv dataset from %s", file_path)

        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not self._is_relevant_arxiv_category(paper.get("categories", "")):
                    continue

                cleaned = self.cleaner.clean_scientific_text(
                    paper.get("abstract", ""),
                    source_type="arxiv",
                )
                if cleaned:
                    texts.append(cleaned)
                    sources.append("arxiv")
                    samples_loaded += 1

                    if self.config.sample_mode and samples_loaded >= self.config.max_samples:
                        break

        return texts, sources

    def load_all_datasets(self) -> Tuple[List[str], List[str]]:
        """Load all configured datasets and return cleaned texts with source labels."""
        cache_fingerprint = {
            "sample_mode": self.config.sample_mode,
            "max_samples": self.config.max_samples,
            "samples_per_source": self.config.samples_per_source,
            "batch_size": self.config.batch_size,
        }

        if self.config.use_cache:
            cache_key = self.cache_manager.make_cache_key(
                base_name="all_datasets",
                config_fingerprint=cache_fingerprint,
            )
            cached = self.cache_manager.load(cache_key)
            if cached is not None:
                self.logger.info("Loaded datasets from cache.")
                return cached

        all_texts: List[str] = []
        all_sources: List[str] = []

        datasets = [
            ("Lay Summaries", self.load_lay_summaries),
            ("Wikipedia", self.load_wikipedia),
            ("SciQ", self.load_sciq),
        ]

        for dataset_name, loader in datasets:
            self.logger.info("Loading %s...", dataset_name)
            texts, sources = loader()
            all_texts.extend(texts)
            all_sources.extend(sources)

        arxiv_path = self.config.base_path / self.config.arxiv_filename
        if arxiv_path.exists():
            self.logger.info("Loading arXiv...")
            texts, sources = self.load_arxiv(arxiv_path)
            all_texts.extend(texts)
            all_sources.extend(sources)

        if self.config.use_cache:
            self.cache_manager.save(
                data=(all_texts, all_sources),
                base_name="all_datasets",
                config_fingerprint=cache_fingerprint,
                extra_metadata={"num_samples": len(all_texts)},
            )

        source_distribution = Counter(all_sources)
        self.logger.info("Total samples loaded: %s", len(all_texts))
        for source, count in source_distribution.items():
            pct = (count / len(all_texts) * 100.0) if all_texts else 0.0
            self.logger.info("%s: %s samples (%.2f%%)", source, count, pct)

        return all_texts, all_sources