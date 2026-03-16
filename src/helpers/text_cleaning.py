"""
Helper functions for scientific text cleaning and filtering.

This module provides source-aware cleaning utilities for scientific text from
datasets such as arXiv, Wikipedia, lay summaries, and SciQ. The cleaning
pipeline is designed to reduce formatting noise while preserving as much
scientific meaning as possible.

It is designed to be imported from the data pipeline rather than executed
directly.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
import contractions


LOGGER = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for scientific text cleaning."""

    normalize_unicode: bool = True
    fix_contractions: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    replace_latex_equations: bool = True
    remove_citations: bool = True
    collapse_whitespace: bool = True
    drop_empty: bool = True


class ScientificTextCleaner:
    """Cleaner for source-specific scientific text processing."""

    def __init__(self) -> None:
        self.latex_pattern = re.compile(
            r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\}",
            re.DOTALL,
        )
        self.numeric_citation_pattern = re.compile(r"\[(?:\d+(?:,\s*)?)+\]")
        self.author_year_citation_pattern = re.compile(
            r"\([A-Z][A-Za-z\-]+(?:\s+et al\.)?,\s*\d{4}\)"
        )
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")

    def clean_scientific_text(
        self,
        text: str,
        source_type: str,
        clean_config: Optional[Dict[str, bool]] = None,
    ) -> str:
        """Clean scientific text according to source type."""
        config = self._merge_config(source_type, clean_config)

        if not isinstance(text, str) or not text.strip():
            LOGGER.debug("Invalid or empty input text for source_type=%s", source_type)
            return ""

        try:
            if source_type == "arxiv":
                text = self._clean_arxiv_text(text, config)
            elif source_type == "wikipedia":
                text = self._clean_wikipedia_text(text, config)
            elif source_type == "lay_summary":
                text = self._clean_lay_summary_text(text, config)
            elif source_type == "sciq":
                text = self._clean_sciq_text(text, config)
            else:
                LOGGER.warning("Unknown source_type '%s'; applying common cleaning only.", source_type)

            text = self._apply_common_cleaning(text, config)
            return text.strip()

        except Exception as exc:
            LOGGER.exception("Error cleaning %s text: %s", source_type, exc)
            return text.strip() if isinstance(text, str) else ""

    def _clean_arxiv_text(self, text: str, config: Dict[str, bool]) -> str:
        """Specific cleaning for arXiv text."""
        if config.get("replace_latex_equations", True):
            text = self.latex_pattern.sub(" [EQUATION] ", text)

        if config.get("remove_citations", True):
            text = self.numeric_citation_pattern.sub("", text)
            text = self.author_year_citation_pattern.sub("", text)

        return text

    def _clean_wikipedia_text(self, text: str, config: Dict[str, bool]) -> str:
        """Specific cleaning for Wikipedia text."""
        if config.get("remove_html", True):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                soup = BeautifulSoup(text, "html.parser")
                text = soup.get_text(separator=" ")

        if config.get("remove_citations", True):
            text = self.numeric_citation_pattern.sub("", text)

        return text

    def _clean_lay_summary_text(self, text: str, config: Dict[str, bool]) -> str:
        """Specific cleaning for lay summaries."""
        if config.get("remove_citations", True):
            text = self.author_year_citation_pattern.sub("", text)
        return text

    def _clean_sciq_text(self, text: str, config: Dict[str, bool]) -> str:
        """Specific cleaning for SciQ text."""
        text = re.sub(r"^[A-D]\:\s*", "", text)
        return text

    def _apply_common_cleaning(self, text: str, config: Dict[str, bool]) -> str:
        """Apply common cleaning steps."""
        if config.get("normalize_unicode", True):
            text = unicodedata.normalize("NFKC", text)

        if config.get("fix_contractions", True):
            text = contractions.fix(text)

        if config.get("remove_urls", True):
            text = self.url_pattern.sub("", text)

        if config.get("collapse_whitespace", True):
            text = re.sub(r"\s+", " ", text)

        return text

    def _default_config_for_source(self, source_type: str) -> Dict[str, bool]:
        """Return default config for a given source."""
        config = {
            "normalize_unicode": True,
            "fix_contractions": True,
            "remove_urls": True,
            "remove_html": True,
            "replace_latex_equations": False,
            "remove_citations": True,
            "collapse_whitespace": True,
            "drop_empty": True,
        }

        if source_type == "arxiv":
            config["replace_latex_equations"] = True

        return config

    def _merge_config(
        self,
        source_type: str,
        clean_config: Optional[Dict[str, bool]],
    ) -> Dict[str, bool]:
        """Merge caller config with defaults."""
        config = self._default_config_for_source(source_type)
        if clean_config:
            config.update(clean_config)
        return config

    def clean_batch(
        self,
        texts: List[str],
        source_type: str,
        batch_size: int = 64,
        max_workers: int = 1,
    ) -> List[str]:
        """
        Clean a batch of texts.

        max_workers=1 keeps behaviour simple and deterministic by default.
        Increase only if profiling shows a clear benefit.
        """
        if not texts:
            LOGGER.warning("Empty batch received for source_type=%s", source_type)
            return []

        cleaned_texts: List[str] = []

        def clean_one(text: str) -> str:
            return self.clean_scientific_text(text, source_type)

        if max_workers <= 1:
            for text in texts:
                cleaned_texts.append(clean_one(text))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    cleaned_texts.extend(list(executor.map(clean_one, batch)))

        if self._default_config_for_source(source_type).get("drop_empty", True):
            cleaned_texts = [text for text in cleaned_texts if text.strip()]

        LOGGER.info(
            "Completed cleaning for %s: %s/%s retained",
            source_type,
            len(cleaned_texts),
            len(texts),
        )
        return cleaned_texts