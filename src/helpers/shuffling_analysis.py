"""
Dataset shuffling and analysis utilities.

This module provides lightweight, reproducible utilities for:
- deterministic shuffling of texts and sources
- source distribution analysis
- basic text/token length analysis
- optional saving of summaries and plots

It is intended for diagnostics and preprocessing support, not as a core
training component.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for dataset analysis and shuffling."""

    seed: int = 42
    plot_size: Tuple[int, int] = (12, 8)
    output_dir: Optional[Path] = None


class ScientificDatasetAnalyzer:
    """Lightweight analyzer for shuffled scientific datasets."""

    def __init__(self, config: Optional[AnalysisConfig] = None, tokenizer=None) -> None:
        self.config = config or AnalysisConfig()
        self.tokenizer = tokenizer
        self.rng = random.Random(self.config.seed)

    def shuffle_data(
        self,
        texts: List[str],
        sources: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Shuffle texts and sources deterministically."""
        if len(texts) != len(sources):
            raise ValueError("Number of texts and sources must match.")

        paired = list(zip(texts, sources))
        self.rng.shuffle(paired)

        if not paired:
            return [], []

        shuffled_texts, shuffled_sources = zip(*paired)
        return list(shuffled_texts), list(shuffled_sources)

    def analyze_dataset(
        self,
        texts: List[str],
        sources: List[str],
    ) -> Dict[str, Any]:
        """Compute basic dataset diagnostics."""
        if len(texts) != len(sources):
            raise ValueError("Number of texts and sources must match.")

        source_distribution = self._analyze_source_distribution(sources)
        length_analysis = self._analyze_text_lengths_by_source(texts, sources)
        basic_stats = self._compute_basic_stats(texts, sources)

        results = {
            "basic_stats": basic_stats,
            "source_distribution": source_distribution,
            "length_analysis": length_analysis,
        }

        if self.config.output_dir is not None:
            self._save_analysis(results)
            self._create_visualizations(results)

        return results

    def _get_length(self, text: str) -> int:
        """Return token length if tokenizer is available, else word count."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        return len(text.split())

    def _analyze_source_distribution(self, sources: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze source counts and percentages."""
        counts = Counter(sources)
        total = len(sources)

        return {
            "counts": dict(counts),
            "percentages": {
                source: (count / total * 100) if total else 0.0
                for source, count in counts.items()
            },
        }

    def _analyze_text_lengths_by_source(
        self,
        texts: List[str],
        sources: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze text lengths grouped by source."""
        grouped = defaultdict(list)

        for text, source in zip(texts, sources):
            grouped[source].append(self._get_length(text))

        analysis: Dict[str, Dict[str, float]] = {}
        for source, lengths in grouped.items():
            arr = np.array(lengths, dtype=np.float32)
            analysis[source] = {
                "mean_length": float(np.mean(arr)),
                "median_length": float(np.median(arr)),
                "std_length": float(np.std(arr)),
                "min_length": int(np.min(arr)),
                "max_length": int(np.max(arr)),
                "total_tokens": int(np.sum(arr)),
            }

        return analysis

    def _compute_basic_stats(self, texts: List[str], sources: List[str]) -> Dict[str, Any]:
        """Compute basic dataset-wide statistics."""
        lengths = [self._get_length(text) for text in texts]

        if not lengths:
            return {
                "total_examples": 0,
                "total_tokens": 0,
                "unique_sources": 0,
                "token_stats": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0,
                    "max": 0,
                },
            }

        arr = np.array(lengths, dtype=np.float32)
        return {
            "total_examples": len(texts),
            "total_tokens": int(np.sum(arr)),
            "unique_sources": len(set(sources)),
            "token_stats": {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            },
        }

    def _save_analysis(self, results: Dict[str, Any]) -> None:
        """Save analysis results to disk."""
        output_dir = self.config.output_dir
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "analysis_results.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

        summary_lines = [
            "Dataset Analysis Summary",
            "========================",
            "",
            "Basic Statistics:",
            f"Total examples: {results['basic_stats']['total_examples']:,}",
            f"Total tokens: {results['basic_stats']['total_tokens']:,}",
            "",
            "Source Distribution:",
        ]

        for source, count in results["source_distribution"]["counts"].items():
            pct = results["source_distribution"]["percentages"][source]
            summary_lines.append(f"  {source}: {count:,} ({pct:.2f}%)")

        summary_path = output_dir / "analysis_summary.txt"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(summary_lines))

    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create simple matplotlib visualizations."""
        output_dir = self.config.output_dir
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        source_data = results.get("source_distribution", {}).get("percentages", {})
        if source_data:
            plt.figure(figsize=self.config.plot_size)
            plt.bar(list(source_data.keys()), list(source_data.values()))
            plt.title("Distribution of Examples per Source")
            plt.xlabel("Source")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "source_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()

        length_data = results.get("length_analysis", {})
        if length_data:
            means = [stats["mean_length"] for stats in length_data.values()]
            if means:
                plt.figure(figsize=self.config.plot_size)
                plt.hist(means, bins="auto")
                plt.title("Distribution of Mean Text Lengths by Source")
                plt.xlabel("Mean Tokens")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(output_dir / "length_distribution.png", dpi=300, bbox_inches="tight")
                plt.close()