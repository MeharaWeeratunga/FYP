# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Final Year Project (FYP) focused on **Early Prediction of Research Paper Virality in Computer Science**. The project aims to develop machine learning models that can predict paper impact within 30-90 days of publication, significantly improving upon current methods that require 3+ years of citation data.

## Key Research Objectives

- Develop early prediction models (30-90 day timeline vs. current 3+ year requirement)
- Create multi-modal approaches integrating text, network, and temporal data
- Build domain-specific models for computer science publications
- Address bias and fairness issues in prediction evaluation
- Implement explainable AI methods for virality prediction

## Project Architecture

This project involves multiple components:

1. **Data Collection & Processing**: Scripts for gathering data from academic APIs (Semantic Scholar, DBLP, ArXiv)
2. **Feature Engineering**: Multi-modal feature extraction from text, citations, author networks, and metadata
3. **Model Development**: Implementation of transformer-based models (SPECTER, SciBERT), Graph Neural Networks, and ensemble methods
4. **Evaluation Framework**: Bias-aware evaluation with temporal dynamics consideration
5. **Prediction System**: Early prediction pipeline with explainability components

## Common Development Commands

```bash
# Environment setup
pip install -r requirements.txt

# Data processing (when implemented)
python src/data/collect_papers.py
python src/data/preprocess.py

# Model training (when implemented)
python src/models/train_early_predictor.py
python src/models/train_multimodal.py

# Evaluation (when implemented)
python src/evaluation/evaluate_predictions.py
python src/evaluation/bias_analysis.py

# Testing
pytest tests/
```

## Technical Focus Areas

- **Early Prediction**: Models optimized for sparse early citation data
- **Multi-modal Integration**: Combining text embeddings, citation networks, and temporal patterns
- **Computer Science Domain**: Specialized handling of conference vs. journal dynamics
- **Bias Mitigation**: Addressing temporal, venue, author, and geographic biases
- **Explainability**: Causal discovery methods and interpretable predictions

## Performance Benchmarks

Target performance metrics based on state-of-the-art:
- Citation prediction MAE: < 7.35 (current best)
- High-impact classification accuracy: > 85%
- Multi-modal improvement: > 6.4% AUC over single-modal
- Early prediction capability: 30-90 day timeline

## Development Guidelines

- Follow scientific computing best practices with reproducible research principles
- Implement comprehensive logging and experiment tracking
- Use version control for datasets and model checkpoints
- Maintain separate environments for data processing, training, and inference
- Document all hyperparameters and experimental configurations