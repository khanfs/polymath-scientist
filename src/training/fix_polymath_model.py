"""
Utility functions for validating and repairing saved Polymath model checkpoints.

This module provides:
- checkpoint file validation
- copying missing tokenizer/config files from a source directory
- loading and re-saving a checkpoint in a consistent format

Validate and repair incomplete saved checkpoints without changing learned weights.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckpointFixConfig:
    """Configuration for checkpoint repair."""

    base_path: Path = PROJECT_ROOT
    source_model_dir: str = "fine_tuned_model"
    target_model_dir: str = "fine_tuned_model/checkpoint_fixed"
    log_dir: str = "logs"

    required_files: tuple[str, ...] = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )


class PolymathCheckpointFixer:
    """Validate and repair a saved checkpoint directory."""

    def __init__(self, config: Optional[CheckpointFixConfig] = None) -> None:
        self.config = config or CheckpointFixConfig()

        self.source_path = self.config.base_path / self.config.source_model_dir
        self.target_path = self.config.base_path / self.config.target_model_dir
        self.log_path = self.config.base_path / self.config.log_dir

        self.target_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger."""
        logger = logging.getLogger("checkpoint_fixer")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(self.log_path / "checkpoint_fixer.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        return logger

    def validate_source_exists(self) -> None:
        """Check that source checkpoint directory exists."""
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source checkpoint directory not found: {self.source_path}")

    def copy_missing_files(self) -> None:
        """Copy missing required files from source directory to target directory."""
        self.logger.info("Copying missing required files from %s to %s", self.source_path, self.target_path)

        for file_name in self.config.required_files:
            source_file = self.source_path / file_name
            target_file = self.target_path / file_name

            if target_file.exists():
                self.logger.info("File already exists in target: %s", target_file.name)
                continue

            if source_file.exists():
                shutil.copy2(source_file, target_file)
                self.logger.info("Copied %s", file_name)
            else:
                self.logger.warning("Missing required file in source: %s", file_name)

    def load_and_resave_checkpoint(self) -> None:
        """Load model/tokenizer from target path and re-save them consistently."""
        self.logger.info("Loading tokenizer from %s", self.target_path)
        tokenizer = AutoTokenizer.from_pretrained(self.target_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.logger.info("Loading model from %s", self.target_path)
        model = AutoModelForCausalLM.from_pretrained(self.target_path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        self.logger.info("Re-saving repaired checkpoint to %s", self.target_path)
        model.save_pretrained(self.target_path)
        tokenizer.save_pretrained(self.target_path)

    def validate_checkpoint_contents(self) -> list[str]:
        """Return a list of missing required files in the target checkpoint."""
        missing_files = []
        for file_name in self.config.required_files:
            if not (self.target_path / file_name).exists():
                missing_files.append(file_name)
        return missing_files

    def fix(self) -> None:
        """Run the repair pipeline."""
        self.validate_source_exists()
        self.copy_missing_files()

        missing_before_load = self.validate_checkpoint_contents()
        if missing_before_load:
            self.logger.warning(
                "Target checkpoint still missing files before load/resave: %s",
                ", ".join(missing_before_load),
            )

        self.load_and_resave_checkpoint()

        missing_after_fix = self.validate_checkpoint_contents()
        if missing_after_fix:
            self.logger.warning(
                "Checkpoint repair completed, but some files are still missing: %s",
                ", ".join(missing_after_fix),
            )
        else:
            self.logger.info("Checkpoint repair completed successfully.")