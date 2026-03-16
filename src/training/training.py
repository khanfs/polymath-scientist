"""
Training utilities for fine-tuning the DistilGPT2 student model.

This module provides:
- loading train/validation splits
- tokenization for causal language modelling
- Hugging Face Trainer setup
- model fine-tuning and saving

It is designed to be imported from scripts rather than executed directly.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    """Configuration for student model fine-tuning."""

    base_path: Path = PROJECT_ROOT
    splits_dir: str = "data/splits"
    model_output_dir: str = "fine_tuned_model"
    log_dir: str = "logs"
    cache_dir: str = "cache"

    model_name: str = "distilgpt2"
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3
    max_seq_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    use_mps: bool = True


class StudentModelTrainer:
    """Trainer for fine-tuning the DistilGPT2 student model."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

        self.splits_path = self.config.base_path / self.config.splits_dir
        self.model_output_path = self.config.base_path / self.config.model_output_dir
        self.log_path = self.config.base_path / self.config.log_dir
        self.cache_path = self.config.base_path / self.config.cache_dir

        for path in [
            self.splits_path,
            self.model_output_path,
            self.log_path,
            self.cache_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.device = self._get_device()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for training."""
        logger = logging.getLogger("student_training")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(self.log_path / "training.log")
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

    def _get_device(self) -> torch.device:
        """Select the best available device."""
        if (
            self.config.use_mps
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            self.logger.info("Using MPS device.")
            return torch.device("mps")

        if torch.cuda.is_available():
            self.logger.info("Using CUDA device.")
            return torch.device("cuda")

        self.logger.warning("No GPU backend available. Falling back to CPU.")
        return torch.device("cpu")

    def validate_environment(self) -> None:
        """Validate training environment."""
        self.logger.info("Validating environment...")
        vm = psutil.virtual_memory()

        if vm.available < 4 * 1024 * 1024 * 1024:
            warnings.warn(
                "Less than 4GB of RAM available. Training may be unstable.",
                stacklevel=2,
            )

    def load_split(self, split_name: str) -> Dataset:
        """Load a dataset split from disk."""
        split_path = self.splits_path / f"{split_name}_split.json"
        self.logger.info("Loading %s split from %s", split_name, split_path)

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
        elif isinstance(data, list):
            texts = [item["text"] for item in data if isinstance(item, dict) and "text" in item]
        else:
            raise ValueError(
                f"Unexpected split format in {split_path}. "
                "Expected dict with 'texts' key or list of records."
            )

        if not texts:
            raise ValueError(f"No texts found in split file: {split_path}")

        return Dataset.from_dict({"text": texts})

    def load_model_and_tokenizer(self):
        """Load tokenizer and base student model."""
        self.logger.info("Loading tokenizer and model: %s", self.config.model_name)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        return tokenizer, model

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer,
        split_name: str,
    ) -> Dataset:
        """Tokenize dataset for causal language modelling."""
        self.logger.info("Tokenizing %s dataset...", split_name)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=self.config.batch_size,
            desc=f"Tokenizing {split_name} dataset",
            remove_columns=["text"],
        )

        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
        )

        return tokenized_dataset

    def build_training_arguments(self) -> TrainingArguments:
        """Create Hugging Face training arguments."""
        use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        use_fp16 = self.device.type == "cuda" and not use_bf16

        return TrainingArguments(
            output_dir=str(self.model_output_path),
            overwrite_output_dir=True,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(self.log_path),
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_pin_memory=False,
            disable_tqdm=False,
            report_to=[],
            remove_unused_columns=True,
        )

    def train(self) -> None:
        """Run student model fine-tuning."""
        self.validate_environment()

        tokenizer, model = self.load_model_and_tokenizer()

        train_dataset = self.load_split("train")
        val_dataset = self.load_split("val")

        self.logger.info(
            "Training dataset size: %s | Validation dataset size: %s",
            len(train_dataset),
            len(val_dataset),
        )

        tokenized_train = self.tokenize_dataset(train_dataset, tokenizer, "train")
        tokenized_val = self.tokenize_dataset(val_dataset, tokenizer, "validation")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        training_args = self.build_training_arguments()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        self.logger.info("Starting fine-tuning...")
        trainer.train()

        self.logger.info("Saving fine-tuned student model...")
        model.config.use_cache = True
        trainer.save_model()
        tokenizer.save_pretrained(self.model_output_path)
        trainer.save_state()
        self.logger.info("Fine-tuned model saved successfully.")