"""
Knowledge distillation utilities.

This module provides:
- loading of student and teacher models
- tokenization of dataset splits
- per-teacher projection heads for cross-architecture alignment
- representation-level distillation with a cosine alignment loss
- a reusable distillation trainer with LR scheduling and validation

It is designed to be imported from scripts rather than executed directly.

---
Architecture note
-----------------
The student (DistilGPT-2) and teachers (BioBERT, MatSciBERT, PhysBERT) share a
nominal hidden size of 768, but their representations live in incompatible
embedding spaces.  Directly computing MSE or cosine distance between raw
pooled vectors conflates architectural differences with semantic proximity
and produces noisy gradients.

This module addresses this by introducing *learnable projection heads*:

    student_proj  : Linear(student_hidden, projection_dim) + L2-normalise
    teacher_proj_X: Linear(teacher_hidden, projection_dim) + L2-normalise  (one per teacher)

Both sides are projected into a shared ``projection_dim``-dimensional space
before the distillation loss is computed.  Cosine similarity on unit-norm
vectors equals the dot product, giving well-scaled gradients independent of
representation magnitude.

The distillation loss is computed *per teacher* (rather than against the
mean of teacher representations) so that each domain's signal is preserved
and cannot be cancelled by opposing gradients from other domains.

Causal vs bidirectional pooling note
-------------------------------------
DistilGPT-2 is a *causal* (left-to-right) model.  Its mean-pooled hidden
states aggregate representations where each token only attends to preceding
tokens, yielding an asymmetric, position-dependent summary.  BERT-style
teachers are bidirectional, so their mean-pool has more uniform positional
coverage.  This is an inherent cross-architecture asymmetry; the projection
heads are given capacity to learn a mapping that compensates for it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    get_cosine_schedule_with_warmup,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_best_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    base_path: Path = PROJECT_ROOT
    splits_dir: str = "data/splits"
    model_output_dir: str = "fine_tuned_model"
    log_dir: str = "logs"

    # Tokenization / batching
    max_length: int = 512
    batch_size: int = 1
    num_workers: int = 0

    # Optimisation
    learning_rate: float = 2e-5
    epochs: int = 3
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100

    # Loss weights
    alpha_distill: float = 1.0
    alpha_lm: float = 1.0
    # Collapse penalty weight.  Penalises the student projection for converging
    # to a near-constant output (representation collapse), which was observed
    # when using BioBERT and SciBERT teachers (cosine sim ≈ 0.999 for all texts).
    # Set to 0.0 to disable.
    alpha_collapse: float = 0.1

    # Projection head dimensions
    # DistilGPT-2 and BERT-style teachers both have hidden_size=768, but
    # they occupy different embedding spaces.  Projection heads map both
    # sides into a shared projection_dim-dimensional space before the
    # distillation loss is computed.
    student_hidden_size: int = 768
    teacher_hidden_size: int = 768
    projection_dim: int = 256

    device: str = get_best_device()

    # Set to N to resume from checkpoint_epoch_N and skip epochs 1..N.
    # 0 means start from scratch.
    resume_from_epoch: int = 0


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Linear projection with L2 normalisation.

    Maps representations from one embedding space into a shared
    ``output_dim``-dimensional projection space.  L2 normalisation ensures
    that cosine similarity equals the dot product, giving stable,
    magnitude-independent gradients.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise: output shape [batch, output_dim], norm=1."""
        return F.normalize(self.linear(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PolymathDistillationTrainer:
    """Trainer for multi-teacher representation distillation."""

    # Canonical teacher identifiers.
    TEACHER_NAMES: Tuple[str, ...] = ("bio", "chem", "phys")

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        self.config = config or DistillationConfig()
        self.device = torch.device(self.config.device)

        self.splits_path = self.config.base_path / self.config.splits_dir
        self.model_output_path = self.config.base_path / self.config.model_output_dir
        self.log_path = self.config.base_path / self.config.log_dir

        self.model_output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

        # Projection heads — initialised here so they can be moved to device
        # and included in the optimiser before training begins.
        self.student_proj = ProjectionHead(
            self.config.student_hidden_size,
            self.config.projection_dim,
        ).to(self.device)

        self.teacher_projs: Dict[str, ProjectionHead] = {
            name: ProjectionHead(
                self.config.teacher_hidden_size,
                self.config.projection_dim,
            ).to(self.device)
            for name in self.TEACHER_NAMES
        }

        # EMA of the student projection, used by the collapse penalty.
        # Tracks the running mean of student projected representations across
        # batches.  If all batches produce nearly the same projection, the
        # cosine similarity to this EMA will be high — that's the collapse signal.
        self._student_proj_ema: Optional[torch.Tensor] = None
        self._ema_decay: float = 0.995

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("distillation")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(self.log_path / "distillation.log")
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

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_student_model(self):
        """Load the fine-tuned student model and tokenizer."""
        self.logger.info("Loading fine-tuned student model from %s", self.model_output_path)

        tokenizer = AutoTokenizer.from_pretrained(self.model_output_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        student_model = AutoModelForCausalLM.from_pretrained(self.model_output_path)
        student_model.config.pad_token_id = tokenizer.pad_token_id
        student_model.to(self.device)

        self.logger.info("Student model loaded successfully.")
        return tokenizer, student_model

    def load_teacher_models(self) -> Dict[str, Dict[str, object]]:
        """Load teacher encoders and tokenizers.

        Teacher-to-domain mapping
        -------------------------
        bio  → BioBERT v1.2  (dmis-lab/biobert-base-cased-v1.2)
                              Pre-trained on PubMed and PMC biomedical papers.
        chem → MatSciBERT    (m3rg-iitd/matscibert)
                              Pre-trained on materials science literature.
                              Replaces ChemBERTa, which was trained on SMILES
                              molecular structure notation rather than prose.
        phys → PhysBERT      (thellert/physbert)
                              Pre-trained on arXiv physics papers.
                              Replaces SciBERT, which covered all scientific
                              domains rather than physics specifically.

        All three teachers are now domain-specific natural language models,
        each matched to its domain's primary literature source.
        """
        self.logger.info("Loading teacher models and tokenizers...")

        # All three teachers are BERT-based encoders.  We load them explicitly
        # as BertModel rather than using AutoModel because older checkpoints
        # (e.g. dmis-lab/biobert-base-cased-v1.2) lack the model_type key in
        # config.json that AutoModel requires in transformers >= 4.40.
        teachers = {
            "bio": {
                "model": BertModel.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2"
                ).eval(),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2",
                    use_fast=False,
                ),
            },
            "chem": {
                "model": BertModel.from_pretrained(
                    "m3rg-iitd/matscibert"
                ).eval(),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "m3rg-iitd/matscibert",
                    use_fast=False,
                ),
            },
            "phys": {
                "model": BertModel.from_pretrained(
                    "thellert/physbert_cased"
                ).eval(),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "thellert/physbert_cased",
                    use_fast=False,
                ),
            },
        }

        # Teachers are frozen (eval mode, no gradients).  Run them on CPU to
        # avoid MPS numerical precision issues that produce NaN on certain
        # input sequences.  This does not meaningfully slow training since
        # teacher forward passes are the minority of compute time.
        cpu = torch.device("cpu")
        for teacher in teachers.values():
            teacher["model"].to(cpu)
            teacher["device"] = cpu

        self.logger.info("Teacher models loaded on CPU successfully.")
        return teachers

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def load_and_tokenize_split(self, split_name: str, tokenizer) -> Dataset:
        """Load and tokenize a dataset split."""
        split_path = self.splits_path / f"{split_name}_split.json"
        self.logger.info("Loading split from %s", split_path)

        with split_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
        elif isinstance(data, list):
            texts = [item["text"] for item in data]
        else:
            raise ValueError(
                f"Unexpected split format in {split_path}. "
                "Expected dict with 'texts' key or list of items with 'text'."
            )

        dataset = Dataset.from_dict({"text": texts})

        tokenized_data = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
            ),
            batched=True,
            remove_columns=["text"],
        )

        tokenized_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
        )

        return tokenized_data

    def create_dataloader(self, tokenized_dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Wrap tokenized dataset in a DataLoader."""
        return DataLoader(
            tokenized_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    # ------------------------------------------------------------------
    # Representation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool hidden states using an attention mask.

        hidden_states : [batch, seq, hidden]
        attention_mask: [batch, seq]
        returns       : [batch, hidden]
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _prepare_teacher_inputs(
        self,
        input_ids: torch.Tensor,
        student_tokenizer,
        teacher_tokenizer,
        teacher_bundle: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode student input_ids and re-tokenize for a teacher model."""
        decoded_texts = [
            student_tokenizer.decode(ids, skip_special_tokens=True)
            for ids in input_ids
        ]

        teacher_inputs = teacher_tokenizer(
            decoded_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )

        # Route to teacher's device (CPU) rather than the student's device (MPS)
        teacher_device = teacher_bundle.get("device", torch.device("cpu")) if teacher_bundle else torch.device("cpu")
        return {k: v.to(teacher_device) for k, v in teacher_inputs.items()}

    def _compute_teacher_representations(
        self,
        teachers: Dict[str, Dict[str, object]],
        input_ids: torch.Tensor,
        student_tokenizer,
    ) -> Dict[str, torch.Tensor]:
        """
        Run all teacher models and return raw pooled representations.

        Returns dict mapping teacher name → [batch, teacher_hidden_size].
        Teacher forward passes are wrapped in torch.no_grad().
        """
        teacher_representations: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for teacher_name, teacher_bundle in teachers.items():
                teacher_model = teacher_bundle["model"]
                teacher_tokenizer = teacher_bundle["tokenizer"]

                teacher_inputs = self._prepare_teacher_inputs(
                    input_ids=input_ids,
                    student_tokenizer=student_tokenizer,
                    teacher_tokenizer=teacher_tokenizer,
                    teacher_bundle=teacher_bundle,
                )

                outputs = teacher_model(**teacher_inputs)
                pooled = self.masked_mean_pool(
                    outputs.last_hidden_state,
                    teacher_inputs["attention_mask"],
                )
                # Move pooled representation from CPU to the student's device
                # (MPS) for the distillation loss computation.
                teacher_representations[teacher_name] = pooled.to(self.device)

        return teacher_representations

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def collapse_penalty(self, student_projected: torch.Tensor) -> torch.Tensor:
        """
        EMA-based collapse penalty.

        Tracks a running mean (EMA) of the student projection across batches.
        When the current projection is very similar to this running mean, the
        student representations have collapsed — all inputs map to nearly the
        same point.  The penalty is the excess cosine similarity above a safe
        threshold (0.9), which is zero when representations are diverse and
        positive when they collapse.

        This operates in-place on self._student_proj_ema and should only be
        called during training (not evaluation).
        """
        with torch.no_grad():
            proj_detached = student_projected.detach()
            if self._student_proj_ema is None:
                self._student_proj_ema = proj_detached.clone()
            else:
                self._student_proj_ema = (
                    self._ema_decay * self._student_proj_ema
                    + (1.0 - self._ema_decay) * proj_detached
                )
            # Normalise EMA so cosine similarity is well-defined
            ema_norm = F.normalize(self._student_proj_ema, p=2, dim=-1)

        collapse_sim = F.cosine_similarity(
            student_projected, ema_norm.detach(), dim=-1
        ).mean()

        # Only penalise similarity above 0.9 (the safe diversity threshold)
        return torch.clamp(collapse_sim - 0.9, min=0.0)

    @staticmethod
    def representation_distillation_loss(
        student_proj: torch.Tensor,
        teacher_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine alignment loss between projected representations.

        Both inputs are expected to be L2-normalised unit vectors (as produced
        by ``ProjectionHead``), so cosine_similarity equals the dot product
        and the loss lies in [0, 2].
        """
        return (1.0 - F.cosine_similarity(student_proj, teacher_proj, dim=-1)).mean()

    # ------------------------------------------------------------------
    # Optimiser and scheduler
    # ------------------------------------------------------------------

    def _build_optimizer_and_scheduler(
        self,
        student_model: nn.Module,
        total_steps: int,
    ) -> Tuple[torch.optim.Optimizer, object]:
        """
        Build an AdamW optimiser covering the student model and all
        projection heads, plus a cosine-with-warmup LR scheduler.
        """
        params = (
            list(student_model.parameters())
            + list(self.student_proj.parameters())
            + [p for proj in self.teacher_projs.values() for p in proj.parameters()]
        )

        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        student_model,
        student_tokenizer,
        path: Path,
        tag: str = "",
    ) -> None:
        """Save student model, tokenizer, and projection heads to *path*."""
        path.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(path)
        student_tokenizer.save_pretrained(path)

        proj_state = {
            "student_proj": self.student_proj.state_dict(),
            "teacher_projs": {
                name: proj.state_dict()
                for name, proj in self.teacher_projs.items()
            },
        }
        torch.save(proj_state, path / "projection_heads.pt")

        label = f" ({tag})" if tag else ""
        self.logger.info("Saved checkpoint%s to %s", label, path)

    def _load_projection_heads(self, path: Path) -> None:
        """Load projection head weights from a checkpoint directory."""
        proj_path = path / "projection_heads.pt"
        if not proj_path.exists():
            self.logger.warning(
                "No projection_heads.pt found in %s — heads will start from scratch.", path
            )
            return

        state = torch.load(proj_path, map_location=self.device)
        self.student_proj.load_state_dict(state["student_proj"])
        for name, proj in self.teacher_projs.items():
            if name in state["teacher_projs"]:
                proj.load_state_dict(state["teacher_projs"][name])
            else:
                self.logger.warning("No saved state for teacher proj '%s'.", name)

        self.logger.info("Loaded projection heads from %s", proj_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
        self,
        student_model: nn.Module,
        student_tokenizer,
        teachers: Dict[str, Dict[str, object]],
        val_dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate on a validation split.

        Returns avg_distill_loss, avg_lm_loss, avg_total_loss.
        """
        student_model.eval()
        self.student_proj.eval()
        for proj in self.teacher_projs.values():
            proj.eval()

        total_distill = 0.0
        total_lm = 0.0
        total = 0.0
        num_batches = 0

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            if attention_mask.sum() < 2:
                continue

            teacher_reprs = self._compute_teacher_representations(
                teachers=teachers,
                input_ids=input_ids,
                student_tokenizer=student_tokenizer,
            )

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )

            student_hidden = student_outputs.hidden_states[-1]
            student_repr = self.masked_mean_pool(student_hidden, attention_mask)
            student_projected = self.student_proj(student_repr)

            distill_loss = torch.tensor(0.0, device=self.device)
            for teacher_name, teacher_repr in teacher_reprs.items():
                teacher_projected = self.teacher_projs[teacher_name](teacher_repr)
                distill_loss += self.representation_distillation_loss(
                    student_projected, teacher_projected
                )
            distill_loss = distill_loss / len(teacher_reprs)

            lm_loss = student_outputs.loss
            total_loss = (
                self.config.alpha_distill * distill_loss
                + self.config.alpha_lm * lm_loss
            )

            total_distill += distill_loss.item()
            total_lm += lm_loss.item()
            total += total_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "avg_distill_loss": total_distill / n,
            "avg_lm_loss": total_lm / n,
            "avg_total_loss": total / n,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        student_model,
        student_tokenizer,
        teachers: Dict[str, Dict[str, object]],
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Run multi-teacher representation distillation training.

        Parameters
        ----------
        student_model    : fine-tuned DistilGPT-2 to be distilled.
        student_tokenizer: matching tokenizer.
        teachers         : dict returned by ``load_teacher_models()``.
        train_dataset    : tokenized training split.
        val_dataset      : optional tokenized validation split for best-model
                           tracking and per-epoch diagnostics.
        """
        train_loader = self.create_dataloader(train_dataset, shuffle=True)
        val_loader = (
            self.create_dataloader(val_dataset, shuffle=False)
            if val_dataset is not None
            else None
        )

        total_steps = len(train_loader) * self.config.epochs
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            student_model, total_steps
        )

        # Handle checkpoint resumption.
        start_epoch = 0
        best_val_loss = float("inf")

        if self.config.resume_from_epoch > 0:
            resume_ckpt = self.model_output_path / f"checkpoint_epoch_{self.config.resume_from_epoch}"
            if resume_ckpt.exists():
                self.logger.info(
                    "Resuming from epoch %s checkpoint: %s",
                    self.config.resume_from_epoch,
                    resume_ckpt,
                )
                # Reload student weights from the checkpoint into the
                # already-loaded model in-place.
                from transformers import AutoModelForCausalLM
                resumed = AutoModelForCausalLM.from_pretrained(resume_ckpt)
                student_model.load_state_dict(resumed.state_dict())
                del resumed

                self._load_projection_heads(resume_ckpt)

                # Fast-forward the scheduler to match completed steps.
                completed_steps = len(train_loader) * self.config.resume_from_epoch
                for _ in range(completed_steps):
                    scheduler.step()

                start_epoch = self.config.resume_from_epoch
                self.logger.info(
                    "Resumed successfully — starting from epoch %s", start_epoch + 1
                )
            else:
                self.logger.warning(
                    "Resume checkpoint not found at %s — starting from scratch.", resume_ckpt
                )

        for epoch in range(start_epoch, self.config.epochs):
            self.logger.info("Epoch %s/%s", epoch + 1, self.config.epochs)

            student_model.train()
            self.student_proj.train()
            for proj in self.teacher_projs.values():
                proj.train()

            # Reset EMA at the start of each epoch so the collapse detector
            # adapts to the current representation distribution rather than
            # carrying stale state from a previous epoch's distribution.
            self._student_proj_ema = None

            epoch_distill = 0.0
            epoch_lm = 0.0
            epoch_total = 0.0
            epoch_collapse = 0.0
            num_batches = 0
            nan_batches = 0

            # Maintain a rolling weight snapshot so we can recover if NaN
            # propagates into the model weights before being detected.
            # Updated every 500 steps from the last known-good state.
            weight_snapshot = {
                k: v.clone().cpu()
                for k, v in student_model.state_dict().items()
            }
            proj_snapshot = {
                "student": self.student_proj.state_dict(),
                "teachers": {
                    n: p.state_dict() for n, p in self.teacher_projs.items()
                },
            }
            snapshot_step = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Skip batches that are entirely padding — these produce
                # NaN LM loss because all labels get masked to -100.
                if attention_mask.sum() < 2:
                    continue

                optimizer.zero_grad(set_to_none=True)

                # Teacher representations (no grad required)
                teacher_reprs = self._compute_teacher_representations(
                    teachers=teachers,
                    input_ids=input_ids,
                    student_tokenizer=student_tokenizer,
                )

                # Student forward pass.
                # Mask padding positions in labels to -100 so they are
                # excluded from the LM loss.  Without this, pad tokens
                # (which share the eos_token ID in DistilGPT-2) contribute
                # undefined loss values that accumulate into NaN.
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )

                student_hidden = student_outputs.hidden_states[-1]
                student_repr = self.masked_mean_pool(student_hidden, attention_mask)
                student_projected = self.student_proj(student_repr)

                # Per-teacher distillation loss (averaged across teachers).
                # Computing a separate loss per teacher preserves each domain's
                # gradient signal independently, rather than letting teachers
                # cancel each other out through representation averaging.
                distill_loss = torch.tensor(0.0, device=self.device)
                for teacher_name, teacher_repr in teacher_reprs.items():
                    teacher_projected = self.teacher_projs[teacher_name](teacher_repr)
                    distill_loss += self.representation_distillation_loss(
                        student_projected, teacher_projected
                    )
                distill_loss = distill_loss / len(teacher_reprs)

                lm_loss = student_outputs.loss

                # Collapse penalty — penalise if all student projections are
                # converging to the same point in the projection space.
                collapse_loss = self.collapse_penalty(student_projected)

                total_loss = (
                    self.config.alpha_distill * distill_loss
                    + self.config.alpha_lm * lm_loss
                    + self.config.alpha_collapse * collapse_loss
                )

                # Guard against NaN/Inf losses.
                if not torch.isfinite(total_loss):
                    # Log the first few NaN batches with token statistics
                    # to help diagnose the root cause.
                    n_real_tokens = attention_mask.sum().item()
                    n_total_tokens = attention_mask.numel()
                    self.logger.warning(
                        "Non-finite loss (distill=%.4f lm=%.4f) — "
                        "real_tokens=%d/%d — skipping batch.",
                        distill_loss.item(),
                        lm_loss.item(),
                        int(n_real_tokens),
                        n_total_tokens,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue

                total_loss.backward()

                # Clip gradients across student model and all projection heads.
                all_params = (
                    list(student_model.parameters())
                    + list(self.student_proj.parameters())
                    + [p for proj in self.teacher_projs.values() for p in proj.parameters()]
                )
                torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip_norm)

                optimizer.step()
                scheduler.step()

                # Check for NaN in student weights after the update.
                # If detected, restore from the last good snapshot.
                any_nan = any(
                    p.isnan().any().item()
                    for p in student_model.parameters()
                )
                if any_nan:
                    nan_batches += 1
                    self.logger.warning(
                        "NaN detected in student weights at step %d — "
                        "restoring from snapshot (step %d).",
                        num_batches,
                        snapshot_step,
                    )
                    student_model.load_state_dict(
                        {k: v.to(self.device) for k, v in weight_snapshot.items()}
                    )
                    self.student_proj.load_state_dict(proj_snapshot["student"])
                    for n, p in self.teacher_projs.items():
                        p.load_state_dict(proj_snapshot["teachers"][n])
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Update snapshot every 500 clean steps.
                if num_batches > 0 and num_batches % 500 == 0:
                    weight_snapshot = {
                        k: v.clone().cpu()
                        for k, v in student_model.state_dict().items()
                    }
                    proj_snapshot = {
                        "student": {
                            k: v.clone() for k, v in self.student_proj.state_dict().items()
                        },
                        "teachers": {
                            n: {k: v.clone() for k, v in p.state_dict().items()}
                            for n, p in self.teacher_projs.items()
                        },
                    }
                    snapshot_step = num_batches

                epoch_distill += distill_loss.item()
                epoch_lm += lm_loss.item()
                epoch_collapse += collapse_loss.item()
                epoch_total += total_loss.item()
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "d_loss": f"{distill_loss.item():.4f}",
                        "lm": f"{lm_loss.item():.4f}",
                        "col": f"{collapse_loss.item():.4f}",
                        "total": f"{total_loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            n = max(num_batches, 1)
            self.logger.info(
                "Epoch %s | train distill=%.4f  lm=%.4f  collapse=%.4f  total=%.4f  "
                "nan_skipped=%d",
                epoch + 1,
                epoch_distill / n,
                epoch_lm / n,
                epoch_collapse / n,
                epoch_total / n,
                nan_batches,
            )

            # Save per-epoch checkpoint
            epoch_ckpt = self.model_output_path / f"checkpoint_epoch_{epoch + 1}"
            self._save_checkpoint(
                student_model, student_tokenizer, epoch_ckpt, tag=f"epoch {epoch + 1}"
            )

            # Validation and best-model tracking
            if val_loader is not None:
                val_metrics = self._evaluate(
                    student_model=student_model,
                    student_tokenizer=student_tokenizer,
                    teachers=teachers,
                    val_dataloader=val_loader,
                )
                self.logger.info(
                    "Epoch %s | val   distill=%.4f  lm=%.4f  total=%.4f",
                    epoch + 1,
                    val_metrics["avg_distill_loss"],
                    val_metrics["avg_lm_loss"],
                    val_metrics["avg_total_loss"],
                )

                if val_metrics["avg_total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["avg_total_loss"]
                    best_ckpt = self.model_output_path / "checkpoint_best"
                    self._save_checkpoint(
                        student_model, student_tokenizer, best_ckpt, tag="best"
                    )
                    self.logger.info(
                        "New best val loss %.4f — saved to %s",
                        best_val_loss,
                        best_ckpt,
                    )

                # Restore training mode after evaluation
                student_model.train()
                self.student_proj.train()
                for proj in self.teacher_projs.values():
                    proj.train()

        self.logger.info("Saving final distilled student model...")
        self._save_checkpoint(
            student_model, student_tokenizer, self.model_output_path, tag="final"
        )
        self.logger.info("Distilled model saved successfully.")
