# Polymath Scientist: System Architecture

## Overview

The Polymath Scientist system is a research prototype designed to explore whether a language model can acquire **transdisciplinary scientific reasoning** through structured training across multiple scientific domains.

The system combines:

* multidisciplinary scientific data
* a general-purpose student language model
* multiple domain-specialised teacher models
* a knowledge distillation framework

The central hypothesis is that **cross-domain knowledge transfer** can enable a student model to integrate scientific knowledge across physics, chemistry, and biology.

---

## Design Principles

The system is built around the following principles:

### 1. Interdisciplinary Integration
Scientific reasoning often spans multiple domains. The system is explicitly designed to integrate knowledge across disciplines rather than isolate them.

### 2. Modular Architecture
Each stage of the pipeline is modular:

- data loading
- dataset preparation
- training
- distillation
- evaluation

This allows independent iteration and experimentation.

### 3. Lightweight Prototyping
The current system uses:

- DistilGPT2 (student)
- domain-specific BERT models (teachers)

This enables rapid experimentation while preserving the core research idea.

### 4. Extensibility
The architecture is designed to scale to:

- larger datasets
- more powerful student models
- additional scientific domains

---

## System Components

The system consists of four primary components:

1. Data Layer  
2. Student Model Training  
3. Teacher Models  
4. Knowledge Distillation Framework  

---

## 1. Data Layer

### Data Sources

The system constructs a multidisciplinary dataset from:

- arXiv (scientific papers)
- PubMed (biomedical literature)
- Wikipedia / Wikimedia (general scientific knowledge)
- SciQ (science question-answer dataset)

### Data Processing Pipeline

The pipeline performs:

- text extraction (title + abstract or core content)
- removal of metadata and noise
- filtering for scientific relevance
- domain balancing across disciplines
- tokenization
- dataset splitting (train / validation / test)

### Output

The result is a **multidisciplinary scientific corpus** suitable for training a general-purpose language model.

---
## 2. Student Model

### Model Choice

The current system uses:

- **DistilGPT2**

This model serves as the base for learning:

- scientific vocabulary
- cross-domain relationships
- generative reasoning patterns

### Training Objective

The student model is:

1. fine-tuned on the multidisciplinary dataset  
2. subsequently refined via knowledge distillation  

### Role in the System

The student model acts as the **integration layer**, where knowledge from multiple domains is combined.

---

## 3. Teacher Models
The system uses domain-specialised models as sources of structured knowledge:

- **BioBERT** → biology
- **ChemBERTa** → chemistry
- **SciBERT** → physics / scientific literature

### Purpose

Teacher models provide:

- domain-specific representations
- structured semantic knowledge
- specialised contextual understanding

### Rationale

Rather than training a single model from scratch, the system leverages **existing domain expertise** embedded in specialised models.

---

## 4. Knowledge Distillation Framework

### Concept

Knowledge distillation is used to transfer knowledge from teacher models to the student model.

In this system, distillation is **multi-teacher**, meaning:

- multiple teacher models contribute to training a single student model

### Process

1. Input data is processed through teacher models  
2. Teacher outputs (representations / logits) are generated  
3. The student model is trained to approximate these outputs  
4. Loss functions combine:
   - language modelling loss
   - distillation loss  

### Objective

The goal is to:

- transfer domain-specific knowledge into the student
- encourage integration across domains
- improve cross-disciplinary reasoning

---

## System Flow

```text
Scientific Data Sources
(arXiv, PubMed, Wikipedia, SciQ)
                |
                v
   Multidisciplinary Data Pipeline
                |
                v
      Student Model Fine-Tuning
            (DistilGPT2)
                |
                v
 Multi-Teacher Knowledge Distillation
   |              |               |
   v              v               v
BioBERT       ChemBERTa        SciBERT
(Biology)     (Chemistry)      (Physics)
                |
                v
      Polymath Scientist Model
```

---

## Training Pipeline

The training process consists of two stages:

**Stage 1: Pretraining / Fine-Tuning**
* train the student model on the multidisciplinary corpus
* objective: learn general scientific language patterns

**Stage 2: Knowledge Distillation**
* transfer structured knowledge from teacher models
* objective: refine domain-specific understanding and integration

---

## Evaluation Strategy (Work in Progress)

Evaluating interdisciplinary reasoning remains an open challenge.

Current and planned approaches include:

* domain-specific evaluation benchmarks
* cross-domain question answering
* reasoning consistency across disciplines
* qualitative analysis of generated outputs

---

## Limitations

This system is an early-stage research prototype.

Key limitations include:

* limited scale compared to frontier models
* early-stage evaluation framework
* incomplete measurement of interdisciplinary reasoning
* reliance on relatively small student model (DistilGPT2)

---

## Future Work
Planned directions include:

* scaling the dataset and model size
* improving distillation objectives
* incorporating additional domains (e.g. mathematics, materials science)
* developing rigorous evaluation benchmarks
* exploring alternative architectures (e.g. mixture-of-experts, modular systems)

---

## Research Direction

This project is part of a broader effort to explore:

* interdisciplinary AI systems
* scientific foundation models
* AI-assisted discovery

The long-term goal is to investigate whether AI systems can support more integrated scientific reasoning across domains.