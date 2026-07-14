# DARE: Towards Video Analytics in Adverse Capture Environments: A Quality-Degradation-Aware Adaptive Encoding Framework


## Introduction

This repository contains the implementation of **DARE**, a degradation-aware adaptive encoding framework for real-time video analytics under bandwidth constraints.

Existing adaptive video analytics systems mainly optimize encoding configurations based on bandwidth and latency requirements, while ignoring the impact of environmental degradation on encoding sensitivity. In degraded scenarios, such as low illumination, noise, blur, and over-exposure, compression artifacts may further reduce visual information and degrade analytics accuracy.

DARE addresses this problem by exploiting degradation-aware encoding sensitivity patterns. It first analyzes the influence of different encoding parameters under various degradation conditions, then constructs degradation-specific encoding policies for efficient online video analytics.

The framework consists of two stages:

- **Offline stage:** 
  - Session-level encoding sensitivity analysis to identify dominant encoding parameters
  - Decision tree construction and video clustering
  - Encoding algorithms training using imitation learning

- **Online stage:**
  - Real-time degradation feature extraction
  - Encoding parameter model selection


<p align="center">
<img src="structure.png" width="850">
</p>


---

# Environment Setup


## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8


Install dependencies:


```bash
pip install -r requirements.txt

# Code Structure

The repository is organized as follows:

## Baselines

`baselines/` contains the implementations of comparison methods.

- `CASVA/`: CASVA implementation.
- `DAO/`: DAO implementation.
- `FHVAC/`: FHVAC implementation.
- `ILCAS/`: ILCAS implementation.
- `LCA/`: LCA implementation.


## Decision Tree Construction

`decision_tree/` implements decision tree construction and video clustering.

- `extract_deg_feature.py`: Extracts degradation-aware features from video sessions.
- `sensitivity.py`: Calculates the sensitivity of encoding parameters for each session.
- `tree_train.py`: Trains the decision tree based on degradation features and sensitivity.
- `apply_tree.py`: Applies the trained decision tree for cluster identification.


## Encoding Parameter Model

`encoding_parameter_model/` implements the encoding decision model training and inference.

- `deg_feats/`: Stores degradation feature data used for model training.
- `utils/`: Utility functions for training and evaluation.
- `DAgger_train.py`: Trains the encoding decision model using DAgger-based imitation learning.
- `env.py`: Defines the training environment.
- `env_fix.py`: Defines the evaluating environment.
- `Expert.py`: Implements the expert policy for generating training labels.
- `network.py`: Defines the neural network architecture.
- `replay_memory.py`: Implements replay memory for training.
- `test.py`: Evaluates the trained encoding decision model.
- `utils.py`: Provides general utility functions.


## Online Decision

`online_switching.py` implements the online adaptive encoding decision process, including degradation feature extraction and encoding model selection.


