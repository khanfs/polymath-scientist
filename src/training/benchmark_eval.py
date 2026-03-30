"""
benchmark_eval.py — Benchmark evaluation on SciQ, BioASQ (subset), and MMLU science subsets.

Evaluates both the distilled Polymath model and the baseline DistilGPT-2 using
perplexity-based multiple-choice scoring: the model assigns each answer choice
a perplexity score and selects the lowest (most likely) answer.

This approach requires no fine-tuning on the benchmark — it is a zero-shot
evaluation of the model's intrinsic scientific knowledge as reflected in its
language modelling capability.

Usage:
    python src/training/benchmark_eval.py
    python src/training/benchmark_eval.py --benchmarks sciq mmlu
    python src/training/benchmark_eval.py --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    model_path: str = str(PROJECT_ROOT / "fine_tuned_model")
    baseline_model: str = "distilgpt2"
    output_dir: str = str(PROJECT_ROOT / "logs" / "benchmark")
    max_samples: int = 500        # per benchmark split — reduce for quick runs
    max_length: int = 256
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmarks: List[str] = field(default_factory=lambda: ["sciq", "mmlu_bio", "mmlu_chem", "mmlu_phys"])
    log_wandb: bool = True
    wandb_project: str = "polymath-scientist"


# ---------------------------------------------------------------------------
# Perplexity-based multiple choice scoring
# ---------------------------------------------------------------------------

def score_answer(
    model,
    tokenizer,
    question: str,
    answer: str,
    support: str = "",
    device: str = "cpu",
    max_length: int = 256,
) -> float:
    """
    Score an answer candidate using perplexity.

    Constructs a prompt of the form:
        [Support text (if available)] Question: {question} Answer: {answer}

    Returns the mean negative log-likelihood (lower = more likely = better).
    """
    context = f"{support.strip()} " if support.strip() else ""
    prompt = f"{context}Question: {question} Answer: {answer}"

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Only compute loss on the answer tokens (not the question prefix)
    # to measure how well the model predicts the answer given the question.
    question_prefix = f"{context}Question: {question} Answer: "
    prefix_enc = tokenizer(
        question_prefix,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    n_prefix = prefix_enc["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:, :n_prefix] = -100        # mask question prefix from loss
    labels[attention_mask == 0] = -100  # mask padding

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    return outputs.loss.item()  # mean NLL — lower is better


def evaluate_multiple_choice(
    model,
    tokenizer,
    examples: List[Dict],
    device: str,
    max_length: int,
    label: str = "",
) -> Dict:
    """
    Evaluate a model on a list of multiple-choice examples.

    Each example must have:
        question: str
        correct_answer: str
        choices: List[str]  (includes the correct answer)
        support: str (optional context)

    Returns accuracy and per-example results.
    """
    correct = 0
    total = 0
    results = []

    for ex in tqdm(examples, desc=f"Eval {label}", leave=False):
        question = ex["question"]
        correct_answer = ex["correct_answer"]
        choices = ex["choices"]
        support = ex.get("support", "")

        scores = {
            choice: score_answer(
                model, tokenizer, question, choice,
                support=support, device=device, max_length=max_length,
            )
            for choice in choices
        }

        predicted = min(scores, key=scores.get)
        is_correct = predicted.strip().lower() == correct_answer.strip().lower()

        correct += int(is_correct)
        total += 1

        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "predicted": predicted,
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_sciq(max_samples: int) -> List[Dict]:
    """Load SciQ test set and format for multiple-choice evaluation."""
    ds = load_dataset("allenai/sciq", split="test")
    examples = []

    for item in ds:
        choices = [
            item["correct_answer"],
            item["distractor1"],
            item["distractor2"],
            item["distractor3"],
        ]
        # Remove empty distractors
        choices = [c for c in choices if c.strip()]

        examples.append({
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "choices": choices,
            "support": item.get("support", ""),
        })

        if len(examples) >= max_samples:
            break

    return examples


def load_mmlu_subset(subject: str, max_samples: int) -> List[Dict]:
    """
    Load an MMLU subject subset.

    Subject strings map to MMLU dataset names:
        mmlu_bio  → high_school_biology
        mmlu_chem → high_school_chemistry
        mmlu_phys → high_school_physics
    """
    subject_map = {
        "mmlu_bio":  "high_school_biology",
        "mmlu_chem": "high_school_chemistry",
        "mmlu_phys": "high_school_physics",
        "mmlu_college_bio":  "college_biology",
        "mmlu_college_chem": "college_chemistry",
        "mmlu_college_phys": "college_physics",
    }

    if subject not in subject_map:
        raise ValueError(f"Unknown MMLU subject: {subject}. Choose from {list(subject_map)}")

    mmlu_name = subject_map[subject]
    # MMLU uses 'test' split for evaluation
    ds = load_dataset("cais/mmlu", mmlu_name, split="test")
    choices_keys = ["A", "B", "C", "D"]

    examples = []
    for item in ds:
        choices = item["choices"]  # list of 4 answer strings
        correct_idx = item["answer"]  # 0-3
        correct_answer = choices[correct_idx]

        examples.append({
            "question": item["question"],
            "correct_answer": correct_answer,
            "choices": choices,
            "support": "",
        })

        if len(examples) >= max_samples:
            break

    return examples


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """Runs benchmark evaluation for both distilled and baseline models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("benchmark_eval")

    def _load_model(self, model_path: str, label: str) -> Tuple:
        self.logger.info("Loading %s from %s", label, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        self.logger.info("%s loaded (%dM params)", label,
                         sum(p.numel() for p in model.parameters()) // 1_000_000)
        return tokenizer, model

    def _load_benchmark(self, name: str) -> Optional[List[Dict]]:
        """Load a benchmark dataset by name."""
        try:
            if name == "sciq":
                self.logger.info("Loading SciQ test set (max %d)", self.config.max_samples)
                return load_sciq(self.config.max_samples)
            elif name.startswith("mmlu_"):
                self.logger.info("Loading MMLU %s (max %d)", name, self.config.max_samples)
                return load_mmlu_subset(name, self.config.max_samples)
            else:
                self.logger.warning("Unknown benchmark: %s", name)
                return None
        except Exception as e:
            self.logger.error("Failed to load %s: %s", name, e)
            return None

    def run(self) -> Dict:
        start = time.time()

        # Load datasets first
        benchmarks = {}
        for name in self.config.benchmarks:
            examples = self._load_benchmark(name)
            if examples:
                benchmarks[name] = examples
                self.logger.info("  %s: %d examples", name, len(examples))

        if not benchmarks:
            self.logger.error("No benchmarks loaded — exiting")
            return {}

        results = {"config": vars(self.config), "benchmarks": {}}

        # Evaluate distilled model
        self.logger.info("\n=== Evaluating distilled Polymath model ===")
        dist_tok, dist_model = self._load_model(self.config.model_path, "Distilled Polymath")

        for bench_name, examples in benchmarks.items():
            self.logger.info("Benchmark: %s (%d examples)", bench_name, len(examples))
            bench_result = evaluate_multiple_choice(
                dist_model, dist_tok, examples,
                device=self.config.device,
                max_length=self.config.max_length,
                label=f"Polymath/{bench_name}",
            )
            results["benchmarks"].setdefault(bench_name, {})["distilled"] = {
                "accuracy": bench_result["accuracy"],
                "correct": bench_result["correct"],
                "total": bench_result["total"],
            }
            self.logger.info(
                "  Polymath %s: %.1f%% (%d/%d)",
                bench_name,
                bench_result["accuracy"] * 100,
                bench_result["correct"],
                bench_result["total"],
            )

        del dist_model

        # Evaluate baseline
        self.logger.info("\n=== Evaluating baseline DistilGPT-2 ===")
        base_tok, base_model = self._load_model(self.config.baseline_model, "Baseline DistilGPT-2")

        for bench_name, examples in benchmarks.items():
            bench_result = evaluate_multiple_choice(
                base_model, base_tok, examples,
                device=self.config.device,
                max_length=self.config.max_length,
                label=f"Baseline/{bench_name}",
            )
            results["benchmarks"][bench_name]["baseline"] = {
                "accuracy": bench_result["accuracy"],
                "correct": bench_result["correct"],
                "total": bench_result["total"],
            }
            self.logger.info(
                "  Baseline %s: %.1f%% (%d/%d)",
                bench_name,
                bench_result["accuracy"] * 100,
                bench_result["correct"],
                bench_result["total"],
            )

        del base_model

        # Print comparison table
        self._print_results(results)

        # Save results
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = self.output_path / f"benchmark_results_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info("\nResults saved to %s", out_path)

        # Log to W&B
        if self.config.log_wandb and WANDB_AVAILABLE:
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    flat = {}
                    for bench, scores in results["benchmarks"].items():
                        if "distilled" in scores:
                            flat[f"benchmark/{bench}_distilled"] = scores["distilled"]["accuracy"]
                        if "baseline" in scores:
                            flat[f"benchmark/{bench}_baseline"] = scores["baseline"]["accuracy"]
                    _wandb.log(flat)
            except Exception:
                pass

        elapsed = time.time() - start
        self.logger.info("Benchmark evaluation complete in %.0fm %.0fs",
                         elapsed // 60, elapsed % 60)
        return results

    def _print_results(self, results: Dict):
        """Print a formatted comparison table."""
        print("\n" + "=" * 65)
        print("  POLYMATH SCIENTIST — BENCHMARK EVALUATION")
        print("=" * 65)
        print(f"  {'Benchmark':<22} {'Baseline':>10} {'Distilled':>10} {'Δ':>8}")
        print("  " + "-" * 60)

        for bench_name, scores in results["benchmarks"].items():
            base_acc = scores.get("baseline", {}).get("accuracy", 0) * 100
            dist_acc = scores.get("distilled", {}).get("accuracy", 0) * 100
            delta = dist_acc - base_acc
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {bench_name:<22} {base_acc:>9.1f}% {dist_acc:>9.1f}% {abs(delta):>6.1f}% {arrow}")

        print("=" * 65)
        print("  Note: Random chance = 25% for 4-choice, 33% for 3-choice")
        print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymath Scientist benchmark evaluation")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["sciq", "mmlu_bio", "mmlu_chem", "mmlu_phys"],
                        help="Benchmarks to run")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max examples per benchmark (default: 500)")
    parser.add_argument("--model-path", type=str,
                        default=str(PROJECT_ROOT / "fine_tuned_model"),
                        help="Path to distilled model")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    config = BenchmarkConfig(
        model_path=args.model_path,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        log_wandb=not args.no_wandb,
    )

    evaluator = BenchmarkEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
