"""
Vocabulary analysis utilities for scientific text corpora.

This module provides diagnostic tools for analyzing tokenizer coverage over a
scientific corpus, including general token statistics, domain-specific token
coverage proxies, and optional output reports and plots.

It is intended for analysis and reporting, not as a required part of the core
training pipeline.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from transformers import PreTrainedTokenizerBase


LOGGER = logging.getLogger(__name__)


@dataclass
class VocabAnalysisConfig:
    """Configuration for vocabulary analysis."""

    sample_size: int = 1_000_000
    min_freq: int = 5
    max_rare_samples: int = 10
    plot_size: Tuple[int, int] = (12, 8)
    scientific_categories: Tuple[str, ...] = ("physics", "chemistry", "biology")
    batch_size: int = 1000
    random_seed: int = 42


class ScientificVocabularyAnalyzer:
    """Diagnostic analyzer for tokenizer coverage over scientific corpora."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[VocabAnalysisConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or VocabAnalysisConfig()
        self.rng = random.Random(self.config.random_seed)
        self.scientific_terms = self._build_scientific_terms()

    def _build_scientific_terms(self) -> Dict[str, Dict[str, set[str]]]:
        """Build scientific term groups for each domain."""
        return {
            "physics": {
                "core": {"quantum", "relativity", "mechanics", "energy"},
                "particles": {"electron", "proton", "neutron", "quark"},
                "concepts": {"force", "mass", "velocity", "momentum"},
                "fields": {"electromagnetic", "gravitational", "nuclear"},
            },
            "chemistry": {
                "core": {"reaction", "molecule", "compound", "solution"},
                "elements": {"hydrogen", "carbon", "oxygen", "nitrogen"},
                "processes": {"oxidation", "reduction", "catalysis"},
                "properties": {"acidic", "basic", "ionic", "covalent"},
            },
            "biology": {
                "core": {"cell", "gene", "protein", "organism"},
                "processes": {"transcription", "translation", "metabolism"},
                "structures": {"membrane", "nucleus", "ribosome"},
                "molecules": {"dna", "rna", "enzyme", "hormone"},
            },
        }

    def token_generator(self, texts: Iterable[List[int]]):
        """Yield flattened token batches from tokenized texts."""
        batch: List[int] = []
        for text in texts:
            batch.extend(text)
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _get_domain_tokens(self, domain: str) -> set[int]:
        """
        Return tokenizer token IDs associated with a domain term list.

        This is a token-level proxy for domain coverage, not a full concept-level
        representation analysis.
        """
        domain_terms = set().union(*self.scientific_terms[domain].values())
        domain_tokens: set[int] = set()

        for term in domain_terms:
            token_ids = self.tokenizer.encode(term, add_special_tokens=False)
            domain_tokens.update(token_ids)

        return domain_tokens

    def analyze_domain_coverage(
        self,
        token_counts: Counter,
        domain: str,
    ) -> Dict[str, float]:
        """Analyze token-level domain coverage."""
        domain_tokens = self._get_domain_tokens(domain)
        covered = domain_tokens.intersection(token_counts.keys())
        coverage = len(covered) / len(domain_tokens) if domain_tokens else 0.0

        return {
            "total_terms": len(domain_tokens),
            "covered_terms": len(covered),
            "coverage": coverage,
        }

    def analyze_vocabulary_coverage(
        self,
        texts: Iterable[List[int]],
        output_dir: Optional[Path] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze tokenizer coverage over tokenized texts."""
        actual_sample_size = sample_size or self.config.sample_size

        token_counts: Counter = Counter()
        total_tokens = 0

        for batch in self.token_generator(texts):
            if total_tokens + len(batch) > actual_sample_size:
                batch = batch[: actual_sample_size - total_tokens]

            token_counts.update(batch)
            total_tokens += len(batch)

            if total_tokens >= actual_sample_size:
                break

        if not token_counts:
            LOGGER.warning("No tokens available for vocabulary analysis.")
            return {}

        vocab_size = len(self.tokenizer)
        unique_tokens = len(token_counts)
        coverage = unique_tokens / vocab_size if vocab_size else 0.0

        domain_coverage = {
            domain: self.analyze_domain_coverage(token_counts, domain)
            for domain in self.config.scientific_categories
        }

        results = {
            "general_stats": {
                "total_tokens": total_tokens,
                "unique_tokens": unique_tokens,
                "vocab_size": vocab_size,
                "coverage": coverage,
            },
            "domain_coverage": domain_coverage,
            "token_distribution": self._analyze_token_distribution(token_counts),
            "common_tokens": self._get_common_tokens(token_counts),
            "rare_tokens": self._get_rare_tokens(token_counts),
        }

        if output_dir is not None:
            self._save_analysis_results(results, output_dir)
            self._plot_distributions(token_counts, output_dir)

        return results

    def _analyze_token_distribution(self, token_counts: Counter) -> Dict[str, float]:
        """Analyze the distribution of token frequencies."""
        counts = np.array(list(token_counts.values()), dtype=np.float32)
        return {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "std": float(np.std(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        }

    def _get_common_tokens(
        self,
        token_counts: Counter,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return the most common tokens and their decoded forms."""
        common_tokens = []
        for token, count in itertools.islice(token_counts.most_common(), n):
            try:
                decoded = self.tokenizer.decode([token], skip_special_tokens=True)
                common_tokens.append(
                    {
                        "token": token,
                        "decoded": decoded,
                        "count": count,
                    }
                )
            except Exception as exc:
                LOGGER.warning("Error decoding token %s: %s", token, exc)
        return common_tokens

    def _get_rare_tokens(self, token_counts: Counter) -> List[Dict[str, Any]]:
        """Return a reproducible sample of rare tokens."""
        rare_tokens = [
            (token, count)
            for token, count in token_counts.items()
            if count < self.config.min_freq
        ]

        if not rare_tokens:
            return []

        sample_n = min(self.config.max_rare_samples, len(rare_tokens))
        sampled = self.rng.sample(rare_tokens, sample_n)

        results = []
        for token, count in sampled:
            try:
                decoded = self.tokenizer.decode([token], skip_special_tokens=True)
                results.append(
                    {
                        "token": token,
                        "decoded": decoded,
                        "count": count,
                    }
                )
            except Exception as exc:
                LOGGER.warning("Error decoding rare token %s: %s", token, exc)

        return results

    def _plot_distributions(self, token_counts: Counter, output_dir: Path) -> None:
        """Save a token-frequency distribution plot."""
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = np.array(list(token_counts.values()), dtype=np.float32)

        plt.figure(figsize=self.config.plot_size)
        bins = np.logspace(
            np.log10(max(1, counts.min())),
            np.log10(max(1, counts.max())),
            50,
        )

        plt.hist(counts, bins=bins, alpha=0.7)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.title("Token Frequency Distribution")
        plt.xlabel("Token Frequency (log scale)")
        plt.ylabel("Number of Tokens (log scale)")
        plt.tight_layout()
        plt.savefig(output_dir / "token_distribution.png")
        plt.close()

    def _save_analysis_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save analysis results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "vocabulary_analysis.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

        summary_path = output_dir / "vocabulary_analysis_summary.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(self._generate_summary_lines(results)))

    def _generate_summary_lines(self, results: Dict[str, Any]) -> List[str]:
        """Generate a human-readable summary."""
        lines = [
            "Vocabulary Coverage Analysis Summary",
            "===================================",
            "",
            "General Statistics:",
            f"Total tokens analyzed: {results['general_stats']['total_tokens']:,}",
            f"Unique tokens: {results['general_stats']['unique_tokens']:,}",
            f"Vocabulary coverage: {results['general_stats']['coverage']:.2%}",
            "",
            "Domain Coverage:",
        ]

        for domain, stats in results["domain_coverage"].items():
            lines.extend(
                [
                    "",
                    f"{domain.capitalize()}:",
                    f"  Terms analyzed: {stats['total_terms']}",
                    f"  Terms covered: {stats['covered_terms']}",
                    f"  Coverage: {stats['coverage']:.2%}",
                ]
            )

        return lines