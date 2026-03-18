"""
Evaluation utilities for the fine-tuned or distilled Polymath student model.

This module provides:
- loading a saved causal language model checkpoint
- instruction-wrapped prompt formatting
- new-tokens-only decoding (strips prompt echo)
- post-processing to remove training-data artefacts
- saving generated outputs and metadata to JSON

It is designed to be imported from scripts rather than executed directly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_best_device() -> str:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


@dataclass
class GenerationConfig:
    """Configuration for text generation during evaluation."""

    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.2


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    base_path: Path = PROJECT_ROOT
    model_dir: str = "fine_tuned_model"
    output_dir: str = "logs/evaluation"
    log_dir: str = "logs"
    device: str = get_best_device()

    # Wrap prompts in an instruction template so the model sees a completion
    # pattern it can follow rather than treating the prompt as a document start.
    instruction_template: str = "Scientific explanation:\n{prompt}\nAnswer: "

    generation: GenerationConfig = field(default_factory=GenerationConfig)

    prompt_sets: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "biology": [
                "Explain how transcription factors regulate gene expression.",
                "Describe how cell signalling pathways can influence phenotype.",
            ],
            "chemistry": [
                "Explain how molecular structure influences solubility.",
                "Describe the role of chemical bonding in determining material properties.",
            ],
            "physics": [
                "Explain how entropy relates to phase transitions.",
                "Describe how quantum mechanics differs from classical mechanics.",
            ],
            "cross_domain": [
                "Explain how physical constraints can shape biological function.",
                "Discuss how chemistry mediates the relationship between physics and biology.",
                "Explain how quantum mechanisms could, in principle, affect gene regulation.",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Patterns that are training-data artefacts from PubMed abstracts and
# Wikipedia markup rather than meaningful generated content.
_ARTEFACT_PATTERNS: List[re.Pattern] = [
    re.compile(r'images?\s*figure\s*\d+[\.\s]?', re.IGNORECASE),
    re.compile(r'\bfig(?:ure)?\s*\d+[\.\s]', re.IGNORECASE),
    re.compile(r'materials\s+and\s+methods\s*[:\.]?', re.IGNORECASE),
    re.compile(r'methods?\s*[:\.]', re.IGNORECASE),
    re.compile(r'results?\s*[:\.]', re.IGNORECASE),
    re.compile(r'conclusion\s*[:\.]?', re.IGNORECASE),
    re.compile(r'aims?\s+and\s+', re.IGNORECASE),
    re.compile(r'\bp\s*[<=>]\s*0\.\d+', re.IGNORECASE),   # p-values
    re.compile(r'\bn\s*=\s*\d+', re.IGNORECASE),            # sample sizes
    re.compile(r'et\s+al\.', re.IGNORECASE),
    re.compile(r'\[\s*\d+\s*\]'),                            # numeric citations
]


def clean_generated_text(text: str) -> str:
    """
    Remove training-data artefacts from generated text.

    Strips PubMed figure references, section headers, statistical
    notation, and other formatting that leaks from the training corpus
    into generated responses.
    """
    for pattern in _ARTEFACT_PATTERNS:
        text = pattern.sub(' ', text)

    # Collapse multiple spaces / newlines introduced by removals
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove leading punctuation artefacts
    text = re.sub(r'^[\s,;:.]+', '', text).strip()

    return text


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class PolymathEvaluator:
    """Reusable evaluator for prompt-based testing of the Polymath student model."""

    def __init__(self, config: Optional[EvaluationConfig] = None) -> None:
        self.config = config or EvaluationConfig()

        self.model_path = self.config.base_path / self.config.model_dir
        self.output_path = self.config.base_path / self.config.output_dir
        self.log_path = self.config.base_path / self.config.log_dir

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()
        self.device = torch.device(self.config.device)

        self.tokenizer = None
        self.model = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("polymath_evaluation")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if logger.handlers:
            return logger

        file_handler = logging.FileHandler(self.log_path / "evaluation.log")
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

    def load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer from the configured checkpoint directory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        self.logger.info("Loading model and tokenizer from %s", self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
        self.model.eval()

        self.logger.info("Model loaded successfully on %s", self.device)

    def _format_prompt(self, prompt: str) -> str:
        """
        Wrap a raw prompt in the instruction template.

        The template signals to the model that a scientific explanation
        is expected, rather than a continuation of an abstract or article.
        """
        return self.config.instruction_template.format(prompt=prompt)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for a single prompt.

        The prompt is wrapped in an instruction template before tokenisation.
        Only the newly generated tokens are decoded — the input prompt echo
        is stripped from the output automatically.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        formatted = self._format_prompt(prompt)

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Record input length so we can strip the prompt echo from the output.
        input_length = inputs["input_ids"].shape[1]

        generation_config = self.config.generation

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the full output sequence and locate the response by searching
        # for the "Answer:" marker in the decoded text.  This is more robust
        # than slicing at input_length token boundaries (BPE tokens can straddle
        # the prompt/response boundary) or exact string prefix matching
        # (tokenizer round-trips are not always identical to the input string).
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the last occurrence of "Answer:" and take everything after it.
        # Using rfind handles the edge case where "Answer:" appears in the
        # prompt topic text as well.
        marker = "Answer: "
        marker_idx = full_output.rfind(marker)
        if marker_idx != -1:
            raw_response = full_output[marker_idx + len(marker):].lstrip()
        else:
            # Fallback: token-slice if marker not found
            new_tokens = outputs[0][input_length:]
            raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).lstrip()

        # Strip BPE boundary fragment.  The first decoded token of the response
        # sometimes encodes a few characters from the preceding context token,
        # leaving a short lowercase fragment at the start (e.g. "e regulation",
        # "l established", "ences External").  These fragments are always:
        #   (a) lowercase  (b) 1-5 characters  (c) followed by a space
        # Detect and remove them before capitalising.
        if raw_response:
            raw_response = raw_response.lstrip()
            if raw_response and raw_response[0].islower():
                first_space = raw_response.find(" ")
                if 0 < first_space <= 5:
                    raw_response = raw_response[first_space + 1:].lstrip()
            if raw_response:
                raw_response = raw_response[0].upper() + raw_response[1:]

        # Remove training-data artefacts from the generated text.
        return clean_generated_text(raw_response)

    def evaluate_prompt_set(self, category: str, prompts: List[str]) -> List[Dict[str, Any]]:
        """Run evaluation for a group of prompts."""
        self.logger.info("Evaluating prompt set: %s (%s prompts)", category, len(prompts))

        results: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts, start=1):
            self.logger.info(
                "Running prompt %s/%s in category '%s'", idx, len(prompts), category
            )
            formatted = self._format_prompt(prompt)
            response = self.generate_response(prompt)

            results.append(
                {
                    "category": category,
                    "prompt_index": idx,
                    "prompt": prompt,
                    "formatted_prompt": formatted,
                    "response": response,
                }
            )

        return results

    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full structured evaluation and return all results."""
        self.load_model_and_tokenizer()

        all_results: List[Dict[str, Any]] = []
        for category, prompts in self.config.prompt_sets.items():
            category_results = self.evaluate_prompt_set(category, prompts)
            all_results.extend(category_results)

        results_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(self.model_path),
            "device": str(self.device),
            "generation_config": asdict(self.config.generation),
            "instruction_template": self.config.instruction_template,
            "prompt_sets": {k: len(v) for k, v in self.config.prompt_sets.items()},
            "results": all_results,
        }

        return results_payload

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save evaluation results to JSON."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"evaluation_results_{timestamp}.json"

        output_file = self.output_path / filename

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

        self.logger.info("Saved evaluation results to %s", output_file)
        return output_file

    def evaluate_and_save(self, filename: Optional[str] = None) -> Path:
        """Run evaluation and save results to disk."""
        results = self.run_evaluation()
        return self.save_results(results, filename=filename)
