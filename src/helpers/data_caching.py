"""
Caching utilities for scientific data pipeline artifacts.

This module provides a simple, deterministic cache for intermediate pipeline
artifacts such as cleaned texts, balanced corpora, and processed splits.

The cache is intended for internal project use only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache storage."""

    cache_dir: Path
    max_cache_size_gb: float = 10.0
    cache_version: str = "1.0"


class ScientificDataCache:
    """Simple deterministic cache for pipeline artifacts."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.cache_dir / "cache_metadata.json"
        if not self.metadata_path.exists():
            self._write_metadata({})

    def _read_metadata(self) -> Dict[str, Any]:
        """Read cache metadata JSON."""
        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write cache metadata JSON."""
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    def _stable_hash(self, payload: Dict[str, Any]) -> str:
        """Create a stable hash from a JSON-serializable payload."""
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.blake2b(serialized.encode("utf-8"), digest_size=20).hexdigest()

    def make_cache_key(
        self,
        base_name: str,
        config_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a deterministic cache key.

        The key depends on:
        - artifact name
        - cache version
        - optional configuration fingerprint
        """
        payload = {
            "base_name": base_name,
            "cache_version": self.config.cache_version,
            "config_fingerprint": config_fingerprint or {},
        }
        return self._stable_hash(payload)

    def save(
        self,
        data: Any,
        base_name: str,
        config_fingerprint: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save an object to cache and return its cache key."""
        cache_key = self.make_cache_key(base_name, config_fingerprint)
        file_path = self.cache_dir / f"{cache_key}.pkl.gz"

        with gzip.open(file_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = self._read_metadata()
        metadata[cache_key] = {
            "base_name": base_name,
            "file": file_path.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache_version": self.config.cache_version,
            "config_fingerprint": config_fingerprint or {},
            "extra_metadata": extra_metadata or {},
        }
        self._write_metadata(metadata)

        LOGGER.info("Saved cache entry '%s' to %s", cache_key, file_path)
        self._enforce_cache_size_limit()
        return cache_key

    def load(self, cache_key: str) -> Optional[Any]:
        """Load an object from cache by key."""
        metadata = self._read_metadata()
        entry = metadata.get(cache_key)

        if not entry:
            LOGGER.info("No cache entry found for key '%s'.", cache_key)
            return None

        file_path = self.cache_dir / entry["file"]
        if not file_path.exists():
            LOGGER.warning("Cache metadata exists but file is missing for key '%s'.", cache_key)
            return None

        with gzip.open(file_path, "rb") as handle:
            data = pickle.load(handle)

        LOGGER.info("Loaded cache entry '%s' from %s", cache_key, file_path)
        return data

    def delete(self, cache_key: str) -> None:
        """Delete a cache entry and its metadata."""
        metadata = self._read_metadata()
        entry = metadata.get(cache_key)

        if not entry:
            return

        file_path = self.cache_dir / entry["file"]
        if file_path.exists():
            file_path.unlink()

        del metadata[cache_key]
        self._write_metadata(metadata)
        LOGGER.info("Deleted cache entry '%s'.", cache_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        metadata = self._read_metadata()
        for entry in metadata.values():
            file_path = self.cache_dir / entry["file"]
            if file_path.exists():
                file_path.unlink()

        self._write_metadata({})
        LOGGER.info("Cleared all cache entries.")

    def _cache_size_bytes(self) -> int:
        """Return total cache size in bytes."""
        return sum(
            path.stat().st_size
            for path in self.cache_dir.glob("*")
            if path.is_file()
        )

    def _enforce_cache_size_limit(self) -> None:
        """Delete oldest entries if cache exceeds configured size."""
        max_bytes = int(self.config.max_cache_size_gb * 1024 * 1024 * 1024)
        if self._cache_size_bytes() <= max_bytes:
            return

        metadata = self._read_metadata()
        ordered_entries = sorted(
            metadata.items(),
            key=lambda item: item[1].get("created_at_utc", ""),
        )

        while self._cache_size_bytes() > max_bytes and ordered_entries:
            oldest_key, _ = ordered_entries.pop(0)
            self.delete(oldest_key)