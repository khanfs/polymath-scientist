"""
Heuristic topic balancing for multidisciplinary scientific text.

This module provides lightweight keyword-based topic classification and dataset
balancing across physics, chemistry, and biology. It is designed as a simple,
transparent balancing utility for the Polymath Scientist data pipeline.

It is intentionally heuristic rather than a full topic model or classifier.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)


@dataclass
class TopicBalancingConfig:
    """Configuration for heuristic topic balancing."""

    random_seed: int = 42
    max_samples_per_category: int = 10000
    min_samples_per_category: int = 100
    min_keyword_matches: int = 1
    balance_other: bool = False


class ScientificTopicBalancer:
    """Keyword-guided balancer for scientific topic distributions."""

    def __init__(self, config: Optional[TopicBalancingConfig] = None) -> None:
        self.config = config or TopicBalancingConfig()
        self.rng = random.Random(self.config.random_seed)
        self.keywords = self._build_keywords()

    def _build_keywords(self) -> Dict[str, set[str]]:
        """Build flattened keyword sets for each topic."""
        return {
            "physics": {
                "quantum", "relativity", "mechanics", "particle", "wave",
                "energy", "mass", "force", "field", "thermodynamics",
                "electromagnetism", "optics", "astrophysics", "physics",
            },
            "chemistry": {
                "molecule", "molecular", "reaction", "compound", "element",
                "atom", "bond", "electron", "ion", "synthesis",
                "purification", "characterization", "organic", "inorganic",
                "analytical", "chemistry",
            },
            "biology": {
                "cell", "cellular", "gene", "genetic", "protein", "dna",
                "rna", "enzyme", "organism", "membrane", "tissue",
                "sequencing", "assay", "microscopy", "biology",
            },
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple regex tokenizer for keyword matching."""
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())

    def classify_topic(self, text: str) -> Tuple[str, float, Dict[str, int]]:
        """
        Classify text into a topic using keyword matches.

        Returns
        -------
        topic : str
            Predicted topic label or 'other'.
        confidence : float
            Heuristic confidence based on score margin.
        raw_scores : dict
            Raw keyword-match counts per topic.
        """
        if not isinstance(text, str) or not text.strip():
            return "other", 0.0, {"physics": 0, "chemistry": 0, "biology": 0}

        tokens = set(self._tokenize(text))
        if not tokens:
            return "other", 0.0, {"physics": 0, "chemistry": 0, "biology": 0}

        scores = {
            topic: sum(1 for keyword in keywords if keyword in tokens)
            for topic, keywords in self.keywords.items()
        }

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_topic, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        if best_score < self.config.min_keyword_matches:
            return "other", 0.0, scores

        confidence = (best_score - second_score) / max(best_score, 1)
        return best_topic, confidence, scores

    def balance_topics(
        self,
        texts: List[str],
        sources: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Balance texts across science topics.

        Returns
        -------
        balanced_texts : list[str]
        balanced_sources : list[str]
        balanced_topics : list[str]
        """
        if len(texts) != len(sources):
            raise ValueError("Number of texts and sources must match.")

        grouped: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

        for text, source in zip(texts, sources):
            topic, confidence, _ = self.classify_topic(text)

            if topic == "other" and not self.config.balance_other:
                continue

            grouped[topic].append((text, source, confidence))

        distribution_before = {topic: len(items) for topic, items in grouped.items()}
        LOGGER.info("Initial topic distribution: %s", distribution_before)

        valid_topics = [
            topic for topic in grouped
            if len(grouped[topic]) >= self.config.min_samples_per_category
        ]

        if not valid_topics:
            LOGGER.warning("No topics met the minimum sample threshold.")
            return [], [], []

        target_size = min(
            min(len(grouped[topic]) for topic in valid_topics),
            self.config.max_samples_per_category,
        )

        balanced_records: List[Tuple[str, str, str]] = []

        for topic in valid_topics:
            items = sorted(grouped[topic], key=lambda item: item[2], reverse=True)
            selected = items[:target_size]
            balanced_records.extend((text, source, topic) for text, source, _ in selected)

        self.rng.shuffle(balanced_records)

        balanced_texts = [record[0] for record in balanced_records]
        balanced_sources = [record[1] for record in balanced_records]
        balanced_topics = [record[2] for record in balanced_records]

        distribution_after = Counter(balanced_topics)
        LOGGER.info("Balanced topic distribution: %s", dict(distribution_after))

        return balanced_texts, balanced_sources, balanced_topics