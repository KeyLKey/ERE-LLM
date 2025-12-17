# ERE-LLM: Entity–Relation Extraction with Large Language Models

This repository contains the official implementation of **ERE-LLM**, a three-stage large language model (LLM) framework for entity–relation extraction in professional domains, as described in our paper:

**ERE-LLM: Entity–Relation Extraction with Large Language Models in Professional Domains**

The framework consists of three stages—**Germination**, **Growth**, and **Maturation**—designed to improve coverage, semantic grounding, and hallucination mitigation in LLM-based information extraction.

---

## Repository Structure

### 1. Germination Stage

- **`Germination.py`**  
  Implements the first stage of ERE-LLM.  
  This stage generates an initial set of entity–relation triples using:
  - Chain-of-Thought (CoT) prompting  
  - In-context learning (ICL)  
  - Multi-perspective dynamic example retrieval  
  - Type-Grounded Prompting for entities and relations  

  The goal of this stage is to produce a semantically grounded and high-recall initialization of candidate triples.

---

### 2. Growth Stage

- **`Growth.py`**  
  Implements the second stage of ERE-LLM.  
  This stage expands the initial triple set by leveraging a predefined database and missed-triple recovery strategies, improving coverage and recall over the Germination output.

---

### 3. Maturation Stage

- **`Maturation-1.py`**  
- **`Maturation-2.py`**  

  These files implement the third stage of ERE-LLM, which focuses on **hallucination mitigation**.  
  The Maturation stage applies dual post-hoc filtering using two lightweight, LoRA-finetuned LLMs:
  - An entity-level relation classification model  
  - A sentence-level relation judgment model  

  Together, they filter out triples that are inconsistent with the source text.

---

## Type-Grounded Prompting

- **`Type-Grounded Prompting with Entities.py`**  
- **`Type-Grounded Prompting with Relationships.py`**  

  These scripts implement **Type-Grounded Prompting** for entities and relations, respectively.  
  Abstract type labels are grounded using data-driven semantic anchors and distilled type/relation definitions, enabling more accurate semantic reasoning in domain-specific settings.

---

## Fine-tuning for Maturation

- **`fine-tuning/`**  

  This directory contains all implementations related to LoRA fine-tuning used in the Maturation stage, including:
  - Model fine-tuning scripts  
  - Training configurations  
  - Data loading and preprocessing utilities  

---

## Type-Constrained Negative Sampling (TCNS)

- **`data_for_Maturation-1.py`**  
- **`data_for_Maturation-2.py`**  

  These scripts implement the **Type-Constrained Negative Sampling (TCNS)** strategy used to construct fine-tuning datasets for the Maturation models.

  Key characteristics:
  - Relation-type constraints are derived from allowed `(head_type, tail_type)` pairs observed in the training data  
  - For each positive sample, one type-compatible negative sample is generated (1:1 ratio)  
  - Negatives are randomly sampled among all type-compatible entity pairs to avoid introducing bias  

  The final generated training datasets are:
  - **`lora_data_Maturation-1.json`**  
  - **`lora_data_Maturation-2.json`**

---

## Notes

- All data used for fine-tuning are derived from open-source resources.
- The code is released for research purposes to support reproducibility and further exploration of LLM-based entity–relation extraction.

---
