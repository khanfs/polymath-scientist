"""
End-to-end pipeline diagnostic.

Run this BEFORE the full pipeline to catch environment and configuration
problems early.  Each stage prints PASS or a clear error message so you
know exactly where to look.

Usage
-----
    python scripts/run_pipeline_check.py

All checks are read-only — nothing is written to disk.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
WARN  = "\033[93m  WARN\033[0m"
INFO  = "\033[94m  INFO\033[0m"

results: list[tuple[str, str, str]] = []   # (stage, status, detail)


def check(label: str):
    """Decorator that runs a check function and records its result."""
    def decorator(fn):
        try:
            detail = fn() or ""
            results.append((label, "PASS", detail))
            print(f"{PASS}  {label}" + (f"\n        {detail}" if detail else ""))
        except Exception as exc:
            detail = str(exc)
            results.append((label, "FAIL", detail))
            print(f"{FAIL}  {label}\n        {detail}")
        return fn
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ──────────────────────────────────────────────────────────────────────────────

@check("Python >= 3.10")
def _():
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (3, 10), f"Python {major}.{minor} found — 3.10+ required"
    return f"Python {major}.{minor}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Third-party packages
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("torch",           "torch"),
    ("transformers",    "transformers"),
    ("datasets",        "datasets"),
    ("accelerate",      "accelerate"),
    ("sklearn",         "scikit-learn"),
    ("numpy",           "numpy"),
    ("tqdm",            "tqdm"),
    ("psutil",          "psutil"),
    ("matplotlib",      "matplotlib"),
    ("bs4",             "beautifulsoup4"),
    ("contractions",    "contractions"),
    ("lxml",            "lxml"),
]

for import_name, pip_name in REQUIRED_PACKAGES:
    @check(f"Package: {pip_name}")
    def _check_pkg(iname=import_name, pname=pip_name):
        mod = importlib.import_module(iname)
        version = getattr(mod, "__version__", "?")
        return f"v{version}"
    _check_pkg()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Internal package imports
# ──────────────────────────────────────────────────────────────────────────────

INTERNAL_MODULES = [
    "src.helpers.text_cleaning",
    "src.helpers.topic_balancing",
    "src.helpers.data_caching",
    "src.helpers.shuffling_analysis",
    "src.helpers.vocabulary_analysis",
    "src.data.load_data",
    "src.data.multidisciplinary_data",
    "src.data.cross_validation",
    "src.data.validate_datasets",
    "src.training.training",
    "src.training.distillation",
    "src.training.evaluate_polymath",
]

for mod_path in INTERNAL_MODULES:
    @check(f"Import: {mod_path}")
    def _check_mod(mp=mod_path):
        importlib.import_module(mp)
    _check_mod()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Device availability
# ──────────────────────────────────────────────────────────────────────────────

@check("PyTorch device")
def _():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        return f"CUDA — {name} ({mem} GB)"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "MPS (Apple Silicon)"
    return "CPU only — training will be slow"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Data splits
# ──────────────────────────────────────────────────────────────────────────────

@check("Split files exist")
def _():
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if not splits_dir.exists():
        raise FileNotFoundError(
            f"data/splits/ not found.\n"
            f"        Run:  python scripts/run_data_pipeline.py"
        )
    found = list(splits_dir.glob("*_split.json"))
    if not found:
        raise FileNotFoundError(
            "No *_split.json files found in data/splits/.\n"
            "        Run:  python scripts/run_data_pipeline.py"
        )
    names = [f.name for f in sorted(found)]
    return f"Found: {names}"


@check("Split files are valid JSON with 'texts' key")
def _():
    import json
    splits_dir = PROJECT_ROOT / "data" / "splits"
    if not splits_dir.exists():
        raise RuntimeError("data/splits/ does not exist — run data pipeline first")

    for split_file in sorted(splits_dir.glob("*_split.json")):
        with split_file.open() as fh:
            data = json.load(fh)
        assert "texts" in data, f"{split_file.name} missing 'texts' key"
        assert isinstance(data["texts"], list), f"'texts' in {split_file.name} is not a list"
        assert len(data["texts"]) > 0, f"{split_file.name} has zero texts"

    counts = {
        f.stem: len(__import__("json").load(f.open())["texts"])
        for f in sorted(splits_dir.glob("*_split.json"))
    }
    return "  ".join(f"{k}={v}" for k, v in counts.items())


# ──────────────────────────────────────────────────────────────────────────────
# 6. Fine-tuned model checkpoint
# ──────────────────────────────────────────────────────────────────────────────

@check("Fine-tuned model checkpoint exists")
def _():
    model_dir = PROJECT_ROOT / "fine_tuned_model"
    if not model_dir.exists():
        raise FileNotFoundError(
            "fine_tuned_model/ not found.\n"
            "        Run:  python scripts/run_training.py"
        )
    required = ["config.json", "tokenizer_config.json"]
    missing  = [f for f in required if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint directory exists but is missing: {missing}\n"
            "        Run:  python scripts/run_training.py  (or fix_polymath_model)"
        )
    # Check for model weights (safetensors or pytorch_model.bin)
    has_weights = (
        list(model_dir.glob("*.safetensors")) or
        list(model_dir.glob("pytorch_model*.bin"))
    )
    if not has_weights:
        raise FileNotFoundError(
            "No model weights found in fine_tuned_model/.\n"
            "        Expected *.safetensors or pytorch_model*.bin"
        )
    return str(model_dir)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Distillation config sanity
# ──────────────────────────────────────────────────────────────────────────────

@check("DistillationConfig instantiates cleanly")
def _():
    from src.training.distillation import DistillationConfig
    cfg = DistillationConfig(base_path=PROJECT_ROOT)
    assert cfg.projection_dim > 0
    assert cfg.alpha_distill >= 0 and cfg.alpha_lm >= 0
    return (
        f"projection_dim={cfg.projection_dim}  "
        f"alpha_distill={cfg.alpha_distill}  "
        f"alpha_lm={cfg.alpha_lm}  "
        f"device={cfg.device}"
    )


@check("ProjectionHead forward pass (no model weights needed)")
def _():
    import torch
    from src.training.distillation import ProjectionHead
    head = ProjectionHead(768, 256)
    x    = torch.randn(2, 768)
    out  = head(x)
    assert out.shape == (2, 256), f"Unexpected shape: {out.shape}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), "Output is not L2-normalised"
    return "shape=(2,256)  L2-norm=1.0  ✓"


@check("masked_mean_pool correctness")
def _():
    import torch
    from src.training.distillation import PolymathDistillationTrainer
    # seq of length 4, last 2 tokens are padding (mask=0)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.9, 9.9], [9.9, 9.9]]])
    mask   = torch.tensor([[1, 1, 0, 0]])
    pooled = PolymathDistillationTrainer.masked_mean_pool(hidden, mask)
    expected = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(pooled, expected, atol=1e-5), f"Got {pooled}, expected {expected}"
    return "Padding correctly ignored  ✓"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Topic balancer smoke test (no network)
# ──────────────────────────────────────────────────────────────────────────────

@check("ScientificTopicBalancer.classify_topic")
def _():
    from src.helpers.topic_balancing import ScientificTopicBalancer
    balancer = ScientificTopicBalancer()

    cases = [
        ("The cell membrane regulates gene expression via protein channels.", "biology"),
        ("The oxidation reaction produced a new ionic compound.", "chemistry"),
        ("Quantum mechanics describes the wave-particle duality of electrons.", "physics"),
    ]
    for text, expected in cases:
        topic, confidence, _ = balancer.classify_topic(text)
        assert topic == expected, f"Expected '{expected}', got '{topic}' for: {text[:50]}"

    return f"All {len(cases)} domain classification checks passed  ✓"


# ──────────────────────────────────────────────────────────────────────────────
# 9. Text cleaner smoke test (no network)
# ──────────────────────────────────────────────────────────────────────────────

@check("ScientificTextCleaner.clean_scientific_text")
def _():
    from src.helpers.text_cleaning import ScientificTextCleaner
    cleaner = ScientificTextCleaner()

    raw_arxiv = "The Hamiltonian $H = p^2/2m$ [1,2,3] describes the system. See https://arxiv.org."
    cleaned   = cleaner.clean_scientific_text(raw_arxiv, source_type="arxiv")
    assert "https://" not in cleaned, "URL not removed"
    assert "[1,2,3]" not in cleaned,  "Citation not removed"
    assert "[EQUATION]" in cleaned,   "LaTeX not replaced"

    return f"arxiv cleaning OK  ✓  ({len(raw_arxiv)}→{len(cleaned)} chars)"


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary():
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")

    print()
    print("=" * 60)
    print(f"  Diagnostic complete:  {passed} passed  /  {failed} failed")
    print("=" * 60)

    if failed:
        print()
        print("  Failed checks:")
        for label, status, detail in results:
            if status == "FAIL":
                print(f"    • {label}")
                print(f"      {detail}")
        print()
        print("  Fix all failures before running the pipeline.")
    else:
        print()
        if not (PROJECT_ROOT / "data" / "splits").exists():
            print("  Next step:  python scripts/run_data_pipeline.py")
        elif not (PROJECT_ROOT / "fine_tuned_model" / "config.json").exists():
            print("  Next step:  python scripts/run_training.py")
        else:
            print("  Next step:  python scripts/run_distillation.py")
    print()


if __name__ == "__main__":
    print()
    print("  Polymath Scientist — End-to-End Diagnostic")
    print("  " + "─" * 50)
    print()
    print_summary()
    failed_count = sum(1 for _, s, _ in results if s == "FAIL")
    sys.exit(1 if failed_count else 0)
