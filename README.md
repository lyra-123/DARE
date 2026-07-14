# DARE: Towards Video Analytics in Adverse Capture Environments: A Quality-Degradation-Aware Adaptive Encoding Framework


## Introduction

This repository contains the implementation of **DARE**, a degradation-aware adaptive encoding framework for real-time video analytics under bandwidth constraints.

Existing adaptive video analytics systems mainly optimize encoding configurations based on bandwidth and latency requirements, while ignoring the impact of environmental degradation on encoding sensitivity. In degraded scenarios, such as low illumination, noise, blur, and over-exposure, compression artifacts may further reduce visual information and degrade analytics accuracy.

DARE addresses this problem by exploiting degradation-aware encoding sensitivity patterns. It first analyzes the influence of different encoding parameters under various degradation conditions, then constructs degradation-specific encoding policies for efficient online video analytics.

The framework consists of two stages:

- **Offline stage:** 
  - Encoding sensitivity analysis
  - Degradation pattern identification using decision trees
  - Encoding policy training using imitation learning

- **Online stage:**
  - Real-time degradation feature extraction
  - Pattern-aware encoding decision selection


<p align="center">
<img src="figures/framework.png" width="850">
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
