# 🎭 Safe-Actor: In-Character Safety & Alignment

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/mahdieh-sjp/XSTest-In-Character-Refusals)
[![GitHub](https://img.shields.io/badge/GitHub-Safe--Actor-black)](https://github.com/mahdieh-sjp/Safe-Actor)

**Training LLMs to maintain persona adherence during roleplay, even when faced with unsafe prompts.**

This repository contains the codebase, generation scripts, and evaluation metrics for the **Safe-Actor** project. Our research addresses the **Persona-Safety Dilemma**—the tendency of aligned Large Language Models (LLMs) to either break character to deliver sterile safety warnings (Failure Mode A) or strictly adhere to a malicious persona and output harmful content (Failure Mode B).

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [The Dataset](#the-dataset)
3. [Model Weights](#model-weights)
4. [Methodology](#methodology)
5. [Repository Structure](#repository-structure)
6. [Evaluation & Results](#evaluation--results)
7. [Usage & Quickstart](#usage--quickstart)
8. [Academic Context](#academic-context)

---

## 🔍 Project Overview

Current models struggle to find the intersection of immersive roleplay and safety alignment. We built a framework that teaches models to enforce both safety and epistemic boundaries **strictly in-character**, eliminating the alignment tax and maintaining complete immersion.

By utilizing **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**, we successfully decoupled this trade-off, allowing models to reject harmful queries using the worldview, tone, and rationale of their assigned persona.

---

## 🗂️ The Dataset

Our human-verified "golden dataset" is fully open-sourced on Hugging Face: 
👉 **[mahdieh-sjp/XSTest-In-Character-Refusals](https://huggingface.co/datasets/mahdieh-sjp/XSTest-In-Character-Refusals)**

### The 7-7-7 Framework
The dataset is built upon queries from the XSTest benchmark and spans 21 diverse personas across three categories:
*   **7 Safe & Wholesome:** (e.g., Doting Grandmother) Focuses on maintaining gentle, nurturing boundaries.
*   **7 Risky & Malicious:** (e.g., Ruthless Cartel Kingpin) Teaches the model to refuse unsafe requests without breaking dark/amoral immersion.
*   **7 Occupational:** (e.g., Criminal Law Professor) Ensures the model can objectively discuss sensitive, in-domain topics without triggering false-positive safety flags.

### Data Splits
*   **Train Set:** 18 personas, covering 80% of the safe and unsafe queries. Formatted for both SFT and contrastive DPO (Chosen vs. Rejected).
*   **Test Set:** The remaining 20% of queries for the 18 training personas, plus **100% of the queries for 3 strictly held-out personas** (Bubbly Baker, Sleazy Corporate Embezzler, Forensic Pathologist) to evaluate zero-shot generalization.

---

## 💾 Model Weights

Due to file size limits, the trained model checkpoints and weights are not hosted directly in this GitHub repository. 

You can download the fine-tuned SFT and DPO models from our public Google Drive folder here:
👉 **[Safe-Actor Models](https://drive.google.com/drive/folders/1CWPhJLvoH8Dj41Rep9pwWGNafvOB-9hA?usp=drive_link)**

---

## 🛠️ Methodology

1.  **Global Initial Generation:** We utilized `Qwen3-4B-Instruct-2507` as the baseline to generate the initial corpus.
2.  **Iterative Refinement:** `Gemini-2.5-Flash-Lite` acted as an LLM-as-a-Judge, strictly flagging character breaks and safety violations. Failed rows were isolated and cyclically regenerated until 100% compliant.
3.  **Human Verification:** A random subset of the final dataset was manually evaluated to guarantee high-quality baselines.
4.  **Alignment Training:** The dataset was used to fine-tune variants using SFT and DPO, actively teaching the model to favor in-character refusals over generic AI disclaimers.

---

## 📊 Evaluation & Results
Our methodology proves that the alignment tax is not an inherent limitation of LLMs.

* **Base Model**: Achieved high safety but sacrificed immersion (Avg Character Score: ~3.4/5).

* **SFT + DPO Variant**: Maintained a perfect safety threshold (blocking 1313/1320 unsafe queries) while successfully raising the character immersion score to ~4.7/5.

* The model successfully generalized the abstract concept of In-Character Safety to the 3 held-out test personas.

---
## 🚀 Usage & Quickstart
**Installation**

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/mahdieh-sjp/Safe-Actor.git
cd Safe-Actor
pip install -r requirements.txt
```

**Loading the Dataset**

You can load the dataset directly from Hugging Face into your pipeline:
```python
from datasets import load_dataset

dataset = load_dataset("mahdieh-sjp/XSTest-In-Character-Refusals")
print(dataset['train'][0])
```

## 🎓 Academic Context
This repository is part of a research project conducted at the University of Vienna, Faculty of Computer Science for Practical Course 2 (July 2026).

* **Researchers**: Mahdieh Sadat Sajedi Pour & Nima Taherianaraki

* **Supervisors**: Prof. Benjamin Roth, Pedro Henrique Luz de Araujo

* **Contact**: mahdiehsajedipoor@gmail.com, n.taheri79@gmail.com

If you use this dataset or codebase in your research, please link back to this repository and our Hugging Face dataset.