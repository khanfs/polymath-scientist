# Polymath Scientist
### A prototype scientific reasoning engine for cross-disciplinary knowledge synthesis

     Scientific Data Sources
(arXiv, Wikipedia, SciQ, eLife, PLoS)
                │
                ▼
           Data Pipeline
(cleaning, topic balancing, validation)
                │
                ▼
          Dataset Splits
        (train / validation)
                │
                ▼
     Student Model Fine-Tuning
            DistilGPT2
                │
                ▼
      Multi-Teacher Distillation
                ▲
                │
   ┌────────────┼────────────┐
   │            │            │
   │            │            │
 BioBERT     ChemBERTa     SciBERT
 (Biology)   (Chemistry)    (Physics)
   │            │            │
   └────────────┴────────────┘
                │
                ▼
       Distilled Polymath Model
                │
                ▼
        Structured Evaluation
     (domain + cross-domain prompts)
                │
                ▼
     Cross-Disciplinary Scientific
            Reasoning Outputs

The system trains a generative student model on multidisciplinary scientific text, then distils representational structure from biology, chemistry, and physics teacher models into a single cross-domain reasoning prototype.

# Polymath Scientist

Polymath Scientist is an experimental scientific reasoning engine prototype that explores how a generative language model can integrate knowledge across multiple scientific domains. The system combines multidisciplinary training of a DistilGPT-2 student model with multi-teacher knowledge distillation from domain-specialized models (BioBERT, ChemBERTa, and SciBERT) to transfer structured expertise from biology, chemistry, and physics into a single generative model. 

The project investigates whether representation-level distillation and cross-domain training can produce a language model capable of synthesizing scientific concepts across disciplines, providing an early step toward AI systems that assist with scientific reasoning, hypothesis generation, and interdisciplinary discovery.

Polymath Scientist

A prototype scientific reasoning engine for cross-disciplinary knowledge synthesis.



Polymath Scientist is an experimental **scientific language model prototype** designed to integrate knowledge across **physics, chemistry, and biology** using **cross-architecture, multi-teacher knowledge distillation**.

The project explores how domain-specialized scientific language models can transfer knowledge to a generalist student model capable of **cross-disciplinary reasoning**.

The long-term goal is to build an **AI system capable of synthesizing insights across scientific domains**, similar to how polymath scientists integrate ideas across fields.

---

# Architecture Overview

This project investigates whether a generative language model can acquire cross-domain scientific reasoning through:

* multidisciplinary fine-tuning of DistilGPT2
* cross-architecture knowledge distillation
* multi-teacher transfer from BioBERT, ChemBERTa, and SciBERT
* custom hybrid architectural components for domain-aware processing and cross-domain interaction

## Models

* **Student model:** DistilGPT2
* **Teacher models:** BioBERT, ChemBERTa, SciBERT

The system consists of three main components.

### Student Model

The student model is a **fine-tuned DistilGPT-2 language model** trained on a multidisciplinary scientific corpus.

It is intended to generate scientific reasoning across domains.

### Teacher Models

Three specialized teacher models provide domain expertise:

| Domain    | Teacher Model |
| --------- | ------------- |
| Biology   | BioBERT       |
| Chemistry | ChemBERTa     |
| Physics   | SciBERT       |

These models are used during **knowledge distillation** to transfer domain-specific knowledge to the student model.

### Knowledge Distillation

The student model learns by minimizing a combined loss:

* **Distillation loss**
  Aligns student predictions with teacher model outputs.

* **Language modelling loss**
  Helps the student maintain coherent text generation.

This produces a student model that integrates signals from multiple scientific teachers.

---

# Project Structure

```text
polymath-scientist/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── multidisciplinary_data.py
│   │   ├── cross_validation.py
│   │   └── validate_datasets.py
│   │
│   ├── helpers/
│   │   ├── text_cleaning.py
│   │   ├── topic_balancing.py
│   │   ├── data_augmentation.py
│   │   ├── data_caching.py
│   │   ├── parallel_processing.py
│   │   ├── vocabulary_analysis.py
│   │   └── shuffling_analysis.py
│   │
│   ├── models/
│   │
│   └── training/
│       └── distillation.py
│
├── scripts/
│   ├── run_data_pipeline.py
│   └── run_training.py
│
├── notebooks/
│   ├── analyse_polymath_dataset.ipynb
│   ├── evaluate_fine_tuned_student_model.ipynb
│   └── fine_tuned_student_model_prompts.ipynb
│
├── data/
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

Generated artifacts such as `splits/`, `fine_tuned_model/`, `logs/`, caches, and large datasets are excluded from version control.

---

# Dataset

The training dataset integrates scientific text from multiple sources:

| Dataset         | Purpose                  |
| --------------- | ------------------------ |
| arXiv           | Research papers          |
| Wikipedia       | Scientific concepts      |
| SciQ            | Science QA dataset       |
| eLife summaries | Biological lay summaries |
| PLoS summaries  | Biomedical lay summaries |

The pipeline is:

```text
raw datasets
      ↓
cleaning
      ↓
topic balancing
      ↓
tokenization
      ↓
dataset analysis
      ↓
train / validation / test splits
```

---

# Running the Data Pipeline

From the project root:

```bash
python scripts/run_data_pipeline.py
```

This will:

1. Load raw datasets
2. Clean and preprocess text
3. Build the multidisciplinary corpus
4. Generate train / validation / test splits
5. Validate the dataset

---

# Training the Distilled Model

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

This will:

1. Load the fine-tuned DistilGPT-2 student model
2. Load teacher models
3. Run multi-teacher knowledge distillation
4. Save the distilled student model

---

# Example Research Questions

This project explores questions such as:

* Can domain-specific scientific language models transfer knowledge to a generalist model?
* Can a student model learn cross-disciplinary scientific reasoning?
* Can multi-teacher distillation produce a polymath-style AI system?

---

# Future Directions

Potential improvements include:

* representation-level distillation
* teacher-to-student projection layers for improved alignment
* cross-modal scientific reasoning
* retrieval-augmented scientific knowledge
* scientific hypothesis generation
* multi-agent scientific reasoning systems

---

# Hardware

The pipeline is designed to run on:

* Apple Silicon (M-series)
* CUDA GPUs
* Google Colab

---

# Status

Research prototype / work in progress.

# Notes

Large datasets, caches, checkpoints, and generated artifacts are excluded from version control.

---

# License

MIT License

---

# Author

**Farooq Khan**

AI researcher working on **transdisciplinary scientific AI systems** and **AI-native discovery platforms**.

---

