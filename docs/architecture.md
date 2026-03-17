# Polymath Scientist: System Architecture

## Overview

Polymath Scientist is a research prototype investigating whether a language model can acquire **interdisciplinary scientific reasoning** through structured training across multiple scientific domains.

The system combines:

- multidisciplinary scientific data  
- a compact autoregressive student language model  
- multiple domain-specialised teacher models  
- a multi-teacher knowledge distillation framework  

The central hypothesis is that **cross-domain knowledge transfer** can enable a student model to integrate scientific knowledge across physics, chemistry, and biology.

## Design Principles

The system is built around the following principles:

### 1. Interdisciplinary Integration
Many important scientific problems emerge at the interface of multiple domains rather than within a single disciplinary silo. The system is therefore designed to integrate knowledge across disciplines rather than isolate them.

### 2. Modular Architecture
Each stage of the pipeline is modular:

- data loading  
- dataset preparation  
- training  
- distillation  
- evaluation  

This enables independent iteration, debugging, and experimentation across system components.

### 3. Lightweight Prototyping
The current system uses:

- DistilGPT2 as the student model  
- domain-specific BERT-family models as teacher models  

This enables rapid experimentation while preserving the core research idea.

### 4. Extensibility
The architecture is designed to scale to:

- larger datasets  
- more capable student models  
- additional scientific domains  
- alternative distillation objectives  
- more modular training strategies  

## System Components

The system consists of four primary components:

1. Data Layer  
2. Student Model Training  
3. Teacher Models  
4. Knowledge Distillation Framework  

## 1. Data Layer

### Data Sources

The system constructs a multidisciplinary dataset from:

- arXiv (scientific papers)  
- PubMed (biomedical literature)  
- Wikipedia / Wikimedia (general scientific knowledge)  
- SciQ (science question-answer dataset)  

### Data Processing Pipeline

The pipeline performs:

- text extraction from titles, abstracts, or core content  
- removal of metadata and noise  
- filtering for scientific relevance  
- balancing of domain coverage  
- tokenization  
- dataset splitting into train / validation / test sets  

### Output

The result is a **multidisciplinary scientific corpus** suitable for adapting a general-purpose language model to scientific text and cross-domain content.

## 2. Student Model

### Model Choice

The current system uses:

- **DistilGPT2**

This model serves as the base for learning:

- scientific vocabulary  
- cross-domain relationships  
- generative scientific language patterns  

### Training Objective

The student model is:

1. fine-tuned on the multidisciplinary dataset  
2. subsequently refined through knowledge distillation  

### Role in the System

The student model functions as the **integration layer** of the system, where knowledge from multiple scientific domains is brought into a shared generative model.

## 3. Teacher Models

The system uses domain-specialised models as sources of structured knowledge:

- **BioBERT** → biology  
- **ChemBERTa** → chemistry  
- **SciBERT** → scientific literature, including physics-oriented scientific text  

### Purpose

Teacher models provide:

- domain-specific representations  
- structured semantic knowledge  
- specialised contextual understanding  

### Rationale

Rather than training a single model from scratch, the system leverages **existing domain expertise** embedded in specialised models and transfers that knowledge into a shared student model.

## 4. Knowledge Distillation Framework

### Concept

Knowledge distillation is used to transfer knowledge from teacher models to the student model.

In this system, distillation is **multi-teacher**, meaning that multiple teacher models contribute to the training of a single student model.

### Process

1. Input data is processed through the teacher models  
2. Teacher signals, such as hidden representations or softened output distributions, are generated  
3. The student model is trained to align with these teacher signals  
4. The training objective combines:
   - language modelling loss  
   - distillation loss  

### Objective

The goal is to:

- transfer domain-specific knowledge into the student model  
- encourage integration across domains  
- improve cross-disciplinary reasoning capacity  

## System Flow

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
## Training Pipeline

The learning dynamics of the system are defined by a two-stage training pipeline that combines domain adaptation with multi-teacher knowledge distillation.

**Stage 1: Domain-Adaptive Fine-Tuning**
* train the student model on the multidisciplinary corpus
* objective: adapt the student model to scientific language and cross-domain content

**Stage 2: Knowledge Distillation**
* transfer structured knowledge from teacher models
* objective: refine domain-specific understanding and support integration across disciplines

## Evaluation Strategy (Work in Progress)

Evaluating interdisciplinary reasoning remains an open challenge.

A key difficulty is distinguishing between:

* domain-specific competence
* cross-domain transfer
* genuinely integrative interdisciplinary reasoning

Current and planned approaches include:

* domain-specific evaluation benchmarks
* cross-domain question answering
* reasoning consistency across disciplines
* qualitative analysis of generated outputs

## Limitations

This system is an early-stage research prototype.

Key limitations include:

* limited scale relative to frontier models
* an early-stage evaluation framework
* incomplete measurement of interdisciplinary reasoning
* reliance on a relatively small student model (DistilGPT2)

## Future Work

Planned directions include:

* scaling the dataset and model size
* improving distillation objectives
* incorporating additional domains such as mathematics and materials science
* developing more rigorous evaluation benchmarks
* exploring alternative architectures, including mixture-of-experts and modular systems

## Research Direction

This project is part of a broader effort to explore:

* interdisciplinary AI systems
* scientific foundation models
* AI-assisted discovery

The long-term aim is to investigate whether AI systems can move beyond narrow domain competence toward more integrated scientific reasoning across multiple fields.