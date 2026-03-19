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
     BioBERT    ChemBERTa    SciBERT
    (Biology)   (Chemistry)  (Scientific Text)
```

The student model is shaped by two complementary learning signals: data-driven adaptation on multidisciplinary scientific text, and teacher-guided distillation from domain-specialised models.

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

* extract scientific content
* reduce noise and metadata
* balance domain coverage
* generate train / validation / test splits
* support lightweight sampling modes for rapid experimentation

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

| Domain          | Teacher Model |
| --------------- | ------------- |
| Biology         | BioBERT       |
| Chemistry       | ChemBERTa     |
| Scientific text | SciBERT       |

These teacher models provide structured representations and domain-specific contextual knowledge that are transferred into the shared student model.

## Knowledge Distillation

The student model learns by minimizing a combined loss:

* **Distillation loss**: Aligns student predictions with teacher model outputs.

* **Language modelling loss**: Helps the student maintain coherent text generation.

These teacher models provide structured representations and domain-specific contextual knowledge that are transferred into the shared student model.

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

## Repository Structure

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

* scientific dataset loading and preprocessing
* multidisciplinary corpus construction
* train / validation / test split generation
* student model training pipeline
* multi-teacher distillation framework
* compatibility fixes for evolving transformer and dataset APIs

### In progress

* evaluation of interdisciplinary reasoning
* stronger benchmark design
* improved distillation objectives
* larger-scale training and ablation studies

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

* interdisciplinary scientific discovery
* hypothesis generation
* scientific literature synthesis
* AI-assisted research workflows
* cross-domain reasoning systems spanning the natural sciences

## Future Directions

Potential improvements include:

* representation-level distillation
* teacher-to-student projection layers for improved alignment
* stronger evaluation of cross-domain scientific reasoning
* retrieval-augmented scientific knowledge
* scientific hypothesis generation
* modular or multi-agent scientific reasoning systems

## Limitations

This is an early-stage research prototype, not a production system.

Current limitations include:

* early evaluation framework
* limited benchmark coverage
* constrained scale relative to frontier foundation models
* ongoing work on measuring true interdisciplinary reasoning

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
AI researcher working on scientific AI, interdisciplinary reasoning, and next-generation AI systems for science.

# License

MIT License