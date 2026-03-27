# AI Polymath Scientist

**AI Polymath Scientist** is a research prototype investigating whether language models can acquire **interdisciplinary scientific reasoning** through multi-teacher knowledge distillation.

The project explores a training framework in which domain-specialised teacher models transfer knowledge from **biology**, **chemistry**, and **physics** into a shared student model. The goal is to study whether this process can support more integrated, cross-domain scientific reasoning.

The system combines multidisciplinary scientific data, domain-adaptive fine-tuning of a generative student model, and multi-teacher distillation into a single cross-domain reasoning prototype.

## Research Motivation

Many AI systems used in science remain domain-specific. In practice, however, important scientific problems often sit at the intersection of multiple fields.

This project explores a different direction: a **polymath-style scientific AI system** trained to integrate knowledge across disciplines rather than operate within a single silo.

The central question is:

> Can a student language model develop broader interdisciplinary scientific reasoning by learning from multiple domain-specialised teacher models?

## Research Questions

This repository investigates questions such as:

1. Can a student model learn cross-domain scientific reasoning?
2. Can multi-teacher distillation improve interdisciplinary knowledge integration in a generative language model?
3. What dataset preparation and training strategies best support cross-domain knowledge transfer?
4. How should interdisciplinary reasoning be evaluated in scientific AI systems?

## System Overview

The system currently consists of three main stages:

1. **Multidisciplinary data preparation**
  Scientific text is collected and processed from multiple sources to create a cross-domain corpus.
2. **Student model training**
  A student language model is adapted to multidisciplinary scientific text.
3. **Multi-teacher knowledge distillation**
  Domain-specialised teacher models transfer structured domain knowledge into the student model.

## Architecture

```text
Scientific Data Sources
(arXiv, PubMed, Wikipedia, SciQ)
                |
                v
   Multidisciplinary Data Pipeline
                |
                v
   Domain-Adaptive Fine-Tuning
         of DistilGPT2
                |
                v
      Polymath Scientist Model
                ^
                |
   Multi-Teacher Knowledge Distillation
        ^           ^           ^
        |           |           |
     BioBERT    ChemBERT     PhysBERT
    (Biology)  (Chemistry)  (Physics)
```

The student model is shaped by two complementary learning signals: data-driven adaptation on multidisciplinary scientific text, and teacher-guided distillation from domain-specialised models.

Distillation uses **six lightweight projection heads** - one student head and one teacher head per domain - mapping each model's representations into a shared 256-dimensional projection space for cosine alignment. Per-domain student heads are necessary: a single shared student head optimised against all three teachers simultaneously collapses to a near-constant output. Per-domain heads allow independent specialisation and prevent cross-domain interference in the projection space.

## Dataset

The training corpus integrates scientific text from multiple sources:


| Dataset         | Purpose                  |
| --------------- | ------------------------ |
| arXiv           | Research papers          |
| Wikipedia       | Scientific concepts      |
| SciQ            | Science QA dataset       |
| eLife summaries | Biological lay summaries |
| PLoS summaries  | Biomedical lay summaries |


## Data Pipeline

The data pipeline is designed to:

- extract scientific content
- reduce noise and metadata
- balance domain coverage
- generate train / validation / test splits
- support lightweight sampling modes for rapid experimentation

**The pipeline is:**

```text
raw datasets
      ↓
text extraction and cleaning
      ↓
topic balancing
      ↓
tokenization
      ↓
dataset analysis
      ↓
train / validation / test splits
```

### Running the Data Pipeline

From the project root:

```bash
python scripts/run_data_pipeline.py
```

This will:

1. Loads raw datasets
2. Cleans and preprocesses text
3. Builds the multidisciplinary corpus
4. Generates train / validation / test splits
5. Validates the dataset outputs

## Models

### Student model

DistilGPT2 is used as the student model. It provides a lightweight autoregressive language model suitable for rapid experimentation in scientific domain adaptation and multi-teacher distillation.

### Teacher Models

Three specialised teacher models provide domain expertise during **knowledge distillation** to transfer domain-specific knowledge to the student model.


| Domain    | Teacher Model                             | Pre-training Data                              | Corpus Match                     |
| --------- | ----------------------------------------- | ---------------------------------------------- | -------------------------------- |
| Biology   | BioBERT v1.2                              | PubMed + PMC biomedical papers                 | ✅ Natural language biology prose |
| Chemistry | ChemBERT (`recobo/chemical-bert-uncased`) | Wikipedia chemistry articles + industrial docs | ✅ Natural language chemistry     |
| Physics   | PhysBERT                                  | 1.2M arXiv physics papers                      | ✅ Natural language physics prose |


Teacher selection is governed by **teacher-corpus alignment**, not just domain proximity. A key empirical finding during development is that domain proximity alone is insufficient: MatSciBERT (materials science literature) was initially used as the chemistry teacher but failed to provide a useful alignment signal for general chemistry text, because materials science prose (alloys, semiconductors, tensile strength) is a genuinely different register from general chemistry prose (molecular solubility, reactions, periodic properties). Despite being chemistry-adjacent, MatSciBERT's pre-training domain did not overlap with the training corpus.

ChemBERT (`recobo/chemical-bert-uncased`) replaces MatSciBERT. It was further pre-trained from SciBERT on Wikipedia chemistry articles and technical chemical documents, directly overlapping with the Wikipedia and SciQ chemistry content in the training corpus. BioBERT and PhysBERT were retained: both are trained on natural language prose in the same domain as the training data, and both produce above-chance domain specificity results.

These teacher models provide structured representations and domain-specific contextual knowledge that are transferred into the shared student model.

## Knowledge Distillation

The student model learns by minimizing a combined loss:

- **Distillation loss**: Aligns student predictions with teacher model outputs via cosine similarity in the shared projection space.
- **Language modelling loss**: Helps the student maintain coherent text generation.

### Frozen Backbone

A key finding during development is that jointly optimising the causal language modelling objective and cross-architecture representational alignment is fundamentally unstable. In practice, distillation gradients caused LM loss to rise from 3.5 to 7–8, progressively overwriting language modelling capability. This is theoretically ill-posed for encoder-decoder distillation pairs, whose attention mechanisms have incompatible geometric structure.

The fix and a core architectural contribution of this project is to **freeze the student backbone entirely** and train only the lightweight per-domain projection heads (~2.4M parameters vs 82M in the full backbone). This produces stable, interpretable domain-specific alignment without degrading the student's generative capability.

### Collapse Detection

During distillation, projection heads were observed to converge to near-constant outputs (cosine similarity ~0.999 across all inputs) - a failure mode where the shared projection space loses discriminative structure. To address this, an **EMA (exponential moving average) collapse penalty** is applied during training. A running mean of the student projection is tracked across batches, and a penalty term is added to the loss when the current projection is too similar to that mean. This mechanism works with `batch_size=1` and directly targets representation collapse in the shared projection space.

### Contrastive Distillation Loss (InfoNCE)

Pure cosine alignment loss produced a **winner-takes-all collapse**: BioBERT dominated all teacher comparisons due to its broad vocabulary compatibility with the training corpus, causing the student to align almost exclusively with the biology teacher regardless of input domain. To address this, the distillation loss was replaced with an **InfoNCE contrastive objective** that explicitly penalises incorrect teacher rankings. Rather than simply minimising distance to all teachers simultaneously, InfoNCE treats the correct domain teacher as a positive example and the other teachers as negatives, forcing the student to discriminate between domains rather than averaging across them.

### Preliminary Results

Results from the frozen backbone + per-domain projection heads + InfoNCE contrastive loss condition:


| Metric                         | Baseline DistilGPT-2 | Distilled Polymath | Improvement   |
| ------------------------------ | -------------------- | ------------------ | ------------- |
| Perplexity (biology)           | 56.9                 | 28.3               | ↓ 28.6        |
| Perplexity (chemistry)         | 56.7                 | 33.6               | ↓ 23.1        |
| Perplexity (physics)           | 53.2                 | 28.6               | ↓ 24.6        |
| Perplexity (overall)           | 55.0                 | 30.1               | ↓ 24.9        |
| Domain specificity (biology)   | —                    | 69.5%              | vs 33% chance |
| Domain specificity (chemistry) | —                    | Below chance       | Open problem  |
| Domain specificity (physics)   | —                    | 48.3%              | vs 33% chance |


Perplexity improvements are strong and consistent across all three domains (~45% overall reduction). Domain specificity is above chance for biology and physics. Chemistry domain specificity remained persistently below chance across all configurations using MatSciBERT, diagnosed as a teacher-corpus mismatch rather than a hyperparameter problem. ChemBERT (`recobo/chemical-bert-uncased`) has been introduced as the new chemistry teacher to address this - results pending.

### Training the Distilled Model

Once dataset splits exist locally, for example:

```text
splits/train_split.json
splits/val_split.json
splits/test_split.json
```

run:

```bash
python scripts/run_training.py
```

This script will:

1. Load the fine-tuned DistilGPT-2 student model
2. Load teacher models
3. Run multi-teacher knowledge distillation
4. Save the distilled student model

## Repository Structure

```text
polymath-scientist/
├── README.md
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── multidisciplinary_data.py
│   │   ├── cross_validation.py
│   │   └── validate_datasets.py
│   ├── training/
│   │   ├── training.py
│   │   └── distillation.py
│   └── ...
├── scripts/
│   └── run_end_to_end.py
├── docs/
│   └── architecture.md
└── ...
```

Generated artifacts such as `splits/`, `fine_tuned_model/`, `logs/`, caches, and large datasets are excluded from version control.

## Current Status

This repository is an active research prototype.

### Implemented

- scientific dataset loading and preprocessing
- multidisciplinary corpus construction
- train / validation / test split generation
- student model training pipeline
- multi-teacher distillation framework
- EMA-based collapse detection and penalty for projection heads
- InfoNCE contrastive distillation loss to prevent winner-takes-all teacher collapse
- Weights & Biases experiment tracking integrated throughout distillation and evaluation
- compatibility fixes for evolving transformer and dataset APIs

### In progress

- evaluation of interdisciplinary reasoning
- stronger benchmark design
- improved distillation objectives
- larger-scale training and ablation studies

## Example Workflow

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the end-to-end pipeline**

```bash
python scripts/run_end_to_end.py
```

## Project Goal

The long-term aim of this project is to explore the development of AI systems capable of more integrated scientific reasoning across domains.
Potential future applications include:

- interdisciplinary scientific discovery
- hypothesis generation
- scientific literature synthesis
- AI-assisted research workflows
- cross-domain reasoning systems spanning the natural sciences

## Future Directions

Potential improvements include:

- scaling to a capable base model (Qwen2.5-1.5B + QLoRA 4-bit quantisation) to test whether the frozen backbone + per-domain projection head approach transfers beyond DistilGPT-2 scale
- representation-level distillation
- teacher-to-student projection layers for improved alignment
- stronger evaluation of cross-domain scientific reasoning
- retrieval-augmented scientific knowledge
- scientific hypothesis generation
- modular or multi-agent scientific reasoning systems

## Limitations

This is an early-stage research prototype, not a production system.

Current limitations include:

- early evaluation framework
- limited benchmark coverage
- constrained scale relative to frontier foundation models
- ongoing work on measuring true interdisciplinary reasoning
- chemistry domain specificity persistently below chance with MatSciBERT — diagnosed as teacher-corpus mismatch, addressed by switching to ChemBERT

## Documentation

- [System Architecture](docs/architecture.md)

## Citation

```bibtex
@misc{khan2026polymathscientist,
  author = {Farooq Khan},
  title = {Polymath Scientist: A Research Prototype for Interdisciplinary Scientific Reasoning},
  year = {2026},
  howpublished = {GitHub repository}
}
```

## Author

Farooq Khan

# License

MIT License