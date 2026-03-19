"""
Quantitative evaluation for the Polymath Scientist distilled model.

This module answers three research questions:

1. PERPLEXITY — Does distillation preserve language modelling quality?
   Compares perplexity of:
   - Base DistilGPT-2 (no training)
   - Distilled Polymath model (fine-tuned + distilled)
   Broken down by domain (biology / chemistry / physics / other) using the
   test split topic labels.

2. REPRESENTATION ALIGNMENT — Did distillation transfer domain knowledge?
   For each teacher (BioBERT, MatSciBERT, PhysBERT), computes the cosine
   similarity between the distilled student's projected representation and
   the teacher's projected representation on test-split texts.
   Higher alignment = more knowledge transferred.

3. DOMAIN SPECIFICITY — Does the student align more with the correct teacher?
   For each domain, checks whether the student representation is closer to
   the domain-matched teacher than to the other two teachers.
   This is the key cross-domain reasoning signal.

All results are saved to logs/evaluation/quantitative_results_<timestamp>.json
and printed as a formatted table.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
)

from src.training.distillation import (
    DistillationConfig,
    PolymathDistillationTrainer,
    ProjectionHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOGGER = logging.getLogger(__name__)


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuantEvalConfig:
    """Configuration for quantitative evaluation."""

    base_path: Path = PROJECT_ROOT
    distilled_model_dir: str = "fine_tuned_model"
    baseline_model_name: str = "distilgpt2"
    splits_dir: str = "data/splits"
    output_dir: str = "logs/evaluation"
    log_dir: str = "logs"

    max_length: int = 512
    batch_size: int = 4
    device: str = get_best_device()

    # Projection head dimensions — must match training config
    student_hidden_size: int = 768
    teacher_hidden_size: int = 768
    projection_dim: int = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("quantitative_eval")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger
    fh = logging.FileHandler(log_path / "quantitative_eval.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
    return logger


def _load_test_split(splits_dir: Path) -> Tuple[List[str], List[str]]:
    """Load test split texts and topic labels."""
    path = splits_dir / "test_split.json"
    if not path.exists():
        raise FileNotFoundError(f"Test split not found: {path}")
    with path.open() as f:
        data = json.load(f)
    texts = data["texts"]
    topics = data.get("topics", ["other"] * len(texts))
    return texts, topics


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Metric 1: Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity_by_domain(
    model,
    tokenizer,
    texts: List[str],
    topics: List[str],
    config: QuantEvalConfig,
    logger: logging.Logger,
    label: str = "model",
) -> Dict[str, float]:
    """
    Compute mean perplexity per domain on the test split.

    Returns dict mapping domain → perplexity.
    """
    model.eval()
    device = torch.device(config.device)

    domain_losses: Dict[str, List[float]] = defaultdict(list)

    logger.info("Computing perplexity for %s on %d texts...", label, len(texts))

    for text, topic in tqdm(
        zip(texts, topics),
        total=len(texts),
        desc=f"Perplexity ({label})",
        leave=False,
    ):
        encoding = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.max_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()

        if math.isfinite(loss):
            domain_losses[topic].append(loss)

    results: Dict[str, float] = {}
    all_losses: List[float] = []

    for domain, losses in sorted(domain_losses.items()):
        ppl = math.exp(sum(losses) / len(losses))
        results[domain] = round(ppl, 3)
        all_losses.extend(losses)

    if all_losses:
        results["overall"] = round(math.exp(sum(all_losses) / len(all_losses)), 3)

    return results


# ---------------------------------------------------------------------------
# Metric 2 & 3: Representation alignment and domain specificity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_alignment_scores(
    student_model,
    student_tokenizer,
    student_proj: ProjectionHead,
    teachers: Dict[str, Dict[str, Any]],
    teacher_projs: Dict[str, ProjectionHead],
    texts: List[str],
    topics: List[str],
    config: QuantEvalConfig,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute cosine alignment between student and each teacher.

    Returns:
        overall_alignment: dict mapping teacher_name → mean cosine similarity
        domain_alignment: dict mapping domain → teacher_name → mean cosine similarity
        domain_specificity: dict mapping domain → fraction of texts where the
            matched teacher has highest alignment
    """
    device = torch.device(config.device)

    student_model.eval()
    student_proj.eval()
    for proj in teacher_projs.values():
        proj.eval()

    # teacher name → domain mapping
    teacher_domain_map = {"bio": "biology", "chem": "chemistry", "phys": "physics"}

    # Accumulate per-text similarity scores
    # shape: {teacher_name: [sim_per_text]}
    all_sims: Dict[str, List[float]] = {name: [] for name in teachers}
    # per domain: {domain: {teacher: [sims]}}
    domain_sims: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {name: [] for name in teachers}
    )

    logger.info("Computing representation alignment on %d texts...", len(texts))

    for text, topic in tqdm(
        zip(texts, topics),
        total=len(texts),
        desc="Alignment",
        leave=False,
    ):
        # Tokenize for student
        student_enc = student_tokenizer(
            text,
            return_tensors="pt",
            max_length=config.max_length,
            truncation=True,
            padding=False,
        )
        s_ids = student_enc["input_ids"].to(device)
        s_mask = student_enc["attention_mask"].to(device)

        if s_ids.shape[1] < 2:
            continue

        # Student representation
        s_out = student_model(
            input_ids=s_ids,
            attention_mask=s_mask,
            output_hidden_states=True,
        )
        s_hidden = s_out.hidden_states[-1]
        s_repr = masked_mean_pool(s_hidden, s_mask)
        s_proj = student_proj(s_repr)   # [1, proj_dim], L2-normalised

        # Teacher representations
        decoded = student_tokenizer.decode(s_ids[0], skip_special_tokens=True)

        for t_name, t_bundle in teachers.items():
            t_model = t_bundle["model"]
            t_tokenizer = t_bundle["tokenizer"]

            t_enc = t_tokenizer(
                decoded,
                return_tensors="pt",
                max_length=config.max_length,
                truncation=True,
                padding=False,
            )
            t_ids = t_enc["input_ids"].to(device)
            t_mask = t_enc["attention_mask"].to(device)

            t_out = t_model(input_ids=t_ids, attention_mask=t_mask)
            t_repr = masked_mean_pool(t_out.last_hidden_state, t_mask)
            t_proj = teacher_projs[t_name](t_repr)  # [1, proj_dim], L2-normalised

            sim = F.cosine_similarity(s_proj, t_proj, dim=-1).item()
            all_sims[t_name].append(sim)
            domain_sims[topic][t_name].append(sim)

    # Overall alignment per teacher
    overall_alignment = {
        t_name: round(sum(sims) / max(len(sims), 1), 4)
        for t_name, sims in all_sims.items()
    }

    # Per-domain alignment
    domain_alignment: Dict[str, Dict[str, float]] = {}
    for domain, teacher_sims in sorted(domain_sims.items()):
        domain_alignment[domain] = {
            t_name: round(sum(sims) / max(len(sims), 1), 4)
            for t_name, sims in teacher_sims.items()
        }

    # Domain specificity: does the student align most with the matched teacher?
    domain_specificity: Dict[str, float] = {}
    for domain, teacher_sims in domain_sims.items():
        # Find which teacher maps to this domain
        matched_teacher = next(
            (t for t, d in teacher_domain_map.items() if d == domain), None
        )
        if matched_teacher is None or not teacher_sims[matched_teacher]:
            continue

        n_texts = len(teacher_sims[matched_teacher])
        n_correct = 0
        for i in range(n_texts):
            text_sims = {t: teacher_sims[t][i] for t in teachers if i < len(teacher_sims[t])}
            if text_sims and max(text_sims, key=text_sims.get) == matched_teacher:
                n_correct += 1

        domain_specificity[domain] = round(n_correct / max(n_texts, 1), 4)

    return {
        "overall_alignment": overall_alignment,
        "domain_alignment": domain_alignment,
        "domain_specificity": domain_specificity,
    }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class QuantitativeEvaluator:
    """Run all three quantitative evaluation metrics."""

    def __init__(self, config: Optional[QuantEvalConfig] = None) -> None:
        self.config = config or QuantEvalConfig()
        self.output_path = self.config.base_path / self.config.output_dir
        self.log_path = self.config.base_path / self.config.log_dir
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logger = _setup_logger(self.log_path)
        self.device = torch.device(self.config.device)

    def _load_distilled_model(self):
        """Load the final distilled student model and tokenizer."""
        path = self.config.base_path / self.config.distilled_model_dir
        self.logger.info("Loading distilled model from %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(path)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(self.device)
        return tokenizer, model

    def _load_baseline_model(self):
        """Load the base DistilGPT-2 model from HuggingFace."""
        self.logger.info("Loading baseline model: %s", self.config.baseline_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.baseline_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.config.baseline_model_name)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(self.device)
        return tokenizer, model

    def _load_projection_heads(self) -> Tuple[ProjectionHead, Dict[str, ProjectionHead]]:
        """Load trained projection heads from the distilled model checkpoint."""
        proj_path = (
            self.config.base_path / self.config.distilled_model_dir / "projection_heads.pt"
        )
        if not proj_path.exists():
            raise FileNotFoundError(f"projection_heads.pt not found at {proj_path}")

        state = torch.load(proj_path, map_location=self.device)

        student_proj = ProjectionHead(
            self.config.student_hidden_size, self.config.projection_dim
        ).to(self.device)
        student_proj.load_state_dict(state["student_proj"])

        teacher_projs = {}
        for name, proj_state in state["teacher_projs"].items():
            proj = ProjectionHead(
                self.config.teacher_hidden_size, self.config.projection_dim
            ).to(self.device)
            proj.load_state_dict(proj_state)
            teacher_projs[name] = proj

        self.logger.info("Loaded projection heads for: %s", list(teacher_projs.keys()))
        return student_proj, teacher_projs

    def _load_teachers(self) -> Dict[str, Dict[str, Any]]:
        """Load teacher models and tokenizers."""
        self.logger.info("Loading teacher models...")
        teachers = {
            "bio": {
                "model": BertModel.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2"
                ).eval().to(self.device),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2", use_fast=False
                ),
            },
            "chem": {
                "model": BertModel.from_pretrained(
                    "m3rg-iitd/matscibert"
                ).eval().to(self.device),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "m3rg-iitd/matscibert", use_fast=False
                ),
            },
            "phys": {
                "model": BertModel.from_pretrained(
                    "thellert/physbert"
                ).eval().to(self.device),
                "tokenizer": AutoTokenizer.from_pretrained(
                    "thellert/physbert", use_fast=False
                ),
            },
        }
        return teachers

    def run(self) -> Dict[str, Any]:
        """Run all quantitative metrics and return results."""
        splits_dir = self.config.base_path / self.config.splits_dir
        texts, topics = _load_test_split(splits_dir)
        self.logger.info("Test split: %d texts across domains: %s",
                         len(texts), sorted(set(topics)))

        # ── Metric 1: Perplexity ──────────────────────────────────────────
        self.logger.info("=== Metric 1: Perplexity ===")

        dist_tokenizer, dist_model = self._load_distilled_model()
        distilled_ppl = compute_perplexity_by_domain(
            dist_model, dist_tokenizer, texts, topics, self.config, self.logger,
            label="distilled"
        )
        self.logger.info("Distilled perplexity: %s", distilled_ppl)
        del dist_model   # free memory before loading baseline

        base_tokenizer, base_model = self._load_baseline_model()
        baseline_ppl = compute_perplexity_by_domain(
            base_model, base_tokenizer, texts, topics, self.config, self.logger,
            label="baseline"
        )
        self.logger.info("Baseline perplexity: %s", baseline_ppl)
        del base_model

        # ── Metrics 2 & 3: Alignment and domain specificity ───────────────
        self.logger.info("=== Metrics 2 & 3: Representation Alignment ===")

        dist_tokenizer2, dist_model2 = self._load_distilled_model()
        student_proj, teacher_projs = self._load_projection_heads()
        teachers = self._load_teachers()

        alignment = compute_alignment_scores(
            student_model=dist_model2,
            student_tokenizer=dist_tokenizer2,
            student_proj=student_proj,
            teachers=teachers,
            teacher_projs=teacher_projs,
            texts=texts,
            topics=topics,
            config=self.config,
            logger=self.logger,
        )

        results = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "n_test_texts": len(texts),
            "domain_counts": {
                domain: topics.count(domain) for domain in sorted(set(topics))
            },
            "perplexity": {
                "baseline_distilgpt2": baseline_ppl,
                "distilled_polymath": distilled_ppl,
                "improvement": {
                    domain: round(baseline_ppl.get(domain, 0) - distilled_ppl.get(domain, 0), 3)
                    for domain in distilled_ppl
                },
            },
            "representation_alignment": alignment,
        }

        return results

    def save_and_print(self, results: Dict[str, Any]) -> Path:
        """Save results to JSON and print a formatted summary table."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.output_path / f"quantitative_results_{timestamp}.json"
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        self.logger.info("Saved quantitative results to %s", out_path)

        self._print_table(results)
        return out_path

    def _print_table(self, results: Dict[str, Any]) -> None:
        """Print a readable summary of all metrics."""
        print()
        print("=" * 65)
        print("  POLYMATH SCIENTIST — QUANTITATIVE EVALUATION")
        print("=" * 65)
        print(f"  Test texts: {results['n_test_texts']}  |  Device: {results['device']}")
        print(f"  Domain counts: {results['domain_counts']}")
        print()

        # Perplexity table
        print("  METRIC 1: Perplexity (lower = better language model)")
        print("  " + "-" * 55)
        print(f"  {'Domain':<15} {'Baseline':>12} {'Distilled':>12} {'Δ (↓ good)':>12}")
        print("  " + "-" * 55)
        baseline = results["perplexity"]["baseline_distilgpt2"]
        distilled = results["perplexity"]["distilled_polymath"]
        improvement = results["perplexity"]["improvement"]
        for domain in sorted(baseline.keys()):
            b = baseline.get(domain, "-")
            d = distilled.get(domain, "-")
            delta = improvement.get(domain, "-")
            arrow = "↓" if isinstance(delta, float) and delta > 0 else ("↑" if isinstance(delta, float) and delta < 0 else "")
            print(f"  {domain:<15} {str(b):>12} {str(d):>12} {f'{delta} {arrow}':>12}")
        print()

        # Alignment table
        align = results["representation_alignment"]
        print("  METRIC 2: Representation Alignment (cosine similarity, higher = better)")
        print("  " + "-" * 55)
        print(f"  {'Teacher':<15} {'Domain':<12} {'Mean Cosine Sim':>16}")
        print("  " + "-" * 55)
        teacher_domain = {"bio": "biology", "chem": "chemistry", "phys": "physics"}
        for t_name, domain in teacher_domain.items():
            sim = align["overall_alignment"].get(t_name, "-")
            print(f"  {t_name:<15} {domain:<12} {str(sim):>16}")
        print()

        # Domain specificity table
        print("  METRIC 3: Domain Specificity (% texts where correct teacher ranks highest)")
        print("  " + "-" * 55)
        print(f"  {'Domain':<15} {'Specificity':>14}  {'Interpretation'}")
        print("  " + "-" * 55)
        for domain, score in sorted(align["domain_specificity"].items()):
            pct = f"{score * 100:.1f}%"
            note = "strong" if score >= 0.5 else ("weak" if score >= 0.33 else "below chance")
            print(f"  {domain:<15} {pct:>14}  {note}")
        print()
        print("  Note: 33% = random chance across 3 teachers")
        print("=" * 65)
        print()

    def evaluate(self) -> Path:
        """Run evaluation and save results."""
        results = self.run()
        return self.save_and_print(results)
