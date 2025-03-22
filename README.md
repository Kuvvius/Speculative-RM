# Speculative Reward Model Boosts Decision Making Ability of LLMs Cost-Effectively

Welcome to the official repository for the paper **"Speculative Reward Model Boosts Decision Making Ability of LLMs Cost-Effectively"**. This anonymous repository contains the implementation, training, and testing scripts for our verification framework, which leverages both BERT and GPT-based architectures to enhance the decision-making capabilities of large language models (LLMs) in structured reasoning tasks (e.g., GSM8K).

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
