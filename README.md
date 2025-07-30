# Speculative Reward Model Boosts Decision Making Ability of LLMs Cost-Effectively

Welcome to the official repository for the paper **"Speculative Reward Model Boosts Decision Making Ability of LLMs Cost-Effectively"**. This anonymous repository contains the implementation, training, and testing scripts for our verification framework, which leverages both BERT and GPT-based architectures to enhance the decision-making capabilities of large language models (LLMs) in structured reasoning tasks (e.g., GSM8K).
[Work in progress]
---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This repository implements a verification framework designed to boost the decision-making ability of LLMs cost-effectively. Key components include:

- **Speculative Reward Modeling:** A reward mechanism that guides LLMs to simulate reasoning trajectories and make better decisions.
- **Multi-Model Architectures:** Implementations based on BERT and GPT models for structured verification and decision-making tasks.
- **Structured Reasoning Tasks:** Focus on tasks involving high-density structured knowledge (e.g., GSM8K), where the model must generate evidence-supported answers.

The repository provides:
- Data models and preprocessing scripts.
- Model architecture definitions.
- Training and testing scripts for GSM8K.
- Utility tools for supporting overall operations.

---

## Installation

Follow these steps to set up your environment:

1. **Create and Activate a Conda Environment:**

   ```bash
   conda create -n srm python=3.10
   conda activate srm
   ```

2. **Install the Required Packages:**

   ```bash
   pip install -r requirement.txt
   ```

---

## Repository Structure

- **`base_data_model.py`**  
  Defines the core data model structures.

- **`base_model.py`**  
  Contains the general model architecture definitions.

- **`base_trainer.py`**  
  Implements the base training framework.

- **`bert_modeling_base.py`**  
  Provides the foundational BERT model implementation.

- **`bert_verifier_data_model.py`**  
  Data model definitions for the BERT-based verifier.

- **`bert_verifier_modeling_gsm8k.py`**  
  BERT-based verifier model implementation for GSM8K.

- **`calculator.py`**  
  A utility script for performing calculations.

- **`data_preprocess.py`**  
  Scripts for data preprocessing and cleaning.

- **`gpt_modeling_base.py`**  
  Provides the foundational GPT model implementation.

- **`verifier_data_model.py`**  
  Data model definitions specific to the verifier.

- **`verifier_modeling_gsm8k.py`**  
  Verifier model implementation for GSM8K using our speculative reward approach.

- **`verifier_training_gsm8k.py`**  
  Training script for the verifier model on GSM8K.

- **`verifier_testing_gsm8k.py`**  
  Testing script for evaluating the verifier model on GSM8K.

- **`train_verifier.sh`** and **`train_verifier_pair.sh`**  
  Shell scripts to facilitate model training.

---

## Usage

### Training

To train the verifier model, you can use one of the following methods:

1. **Using the Shell Script:**

   Execute the following command to start training with paired training data:

   ```bash
   bash train_verifier_pair.sh
   ```

2. **Directly Running the Python Script:**

   Alternatively, run the training script directly:

   ```bash
   python verifier_training_gsm8k.py
   ```

### Testing

After training, evaluate the model by running:

```bash
python3 verifier_testing_gsm8k.py
```

The testing script loads the trained model and evaluates its performance on the GSM8K dataset.
