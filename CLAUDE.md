# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Final Year Project (FYP) focused on **Early Prediction of Research Paper Virality in Computer Science**. The project aims to develop machine learning models that can predict paper impact within 30-90 days of publication, significantly improving upon current methods that require 3+ years of citation data.

## Project Structure

```
FYP/
├── src/                          # Source code
│   ├── data/                     # Data collection and processing
│   │   ├── collectors/           # API clients for data sources
│   │   ├── processors/           # Data preprocessing pipelines
│   │   └── storage/              # Database and caching utilities
│   ├── features/                 # Feature engineering modules
│   │   ├── text_features/        # Text-based feature extraction
│   │   ├── network_features/     # Citation network analysis
│   │   ├── temporal_features/    # Time-based features
│   │   └── social_features/      # Social media and altmetrics
│   ├── models/                   # Machine learning models
│   │   ├── baselines/            # Traditional ML baselines
│   │   ├── transformers/         # Transformer-based models
│   │   ├── graph_models/         # Graph neural networks
│   │   ├── fusion_models/        # Multi-modal fusion
│   │   └── domain_specific/      # CS-specific models
│   ├── evaluation/               # Model evaluation and bias analysis
│   │   ├── bias_detection/       # Systematic bias identification
│   │   ├── fairness_metrics/     # Fairness evaluation metrics
│   │   ├── mitigation_methods/   # Bias mitigation techniques
│   │   └── evaluation_protocols/ # Comprehensive evaluation frameworks
│   ├── prediction/               # Prediction and explainability
│   │   ├── explainers/           # Model interpretation tools
│   │   ├── interactive/          # Interactive prediction systems
│   │   ├── robustness/           # Adversarial robustness
│   │   └── deployment/           # Production deployment
│   └── utils/                    # Utility functions and helpers
├── data/                         # Data storage
│   ├── raw/                      # Raw data from APIs
│   ├── processed/                # Cleaned and preprocessed data
│   └── external/                 # External datasets and embeddings
├── experiments/                  # Experiment tracking
│   ├── logs/                     # Training logs and metrics
│   ├── checkpoints/              # Model checkpoints
│   └── results/                  # Experiment results and analysis
├── notebooks/                    # Jupyter notebooks for exploration
├── docs/                         # Documentation
└── tests/                        # Unit and integration tests
```

## Key Research Focus Areas

### 1. Early Prediction Timeline (30-90 days vs 3+ years)
- Develop models optimized for sparse early citation data
- Handle cold-start problem for newly published papers
- Distinguish "sleeping beauties" from genuinely viral papers

### 2. Multi-Modal Integration
- Combine text embeddings, citation networks, and temporal patterns
- Implement adaptive fusion networks with cross-modal attention
- Integrate social media signals and altmetrics

### 3. Computer Science Domain Specialization
- Model conference vs journal publication dynamics
- Handle CS-specific citation patterns and community effects
- Create subfield-specific models (theory/systems/AI)

### 4. Bias Mitigation and Fairness
- Address temporal, venue, author, and geographic biases
- Implement fairness-aware training and evaluation
- Develop bias detection and mitigation strategies

### 5. Explainable AI
- Provide interpretable insights into virality mechanisms
- Implement causal discovery methods
- Create interactive prediction systems with uncertainty quantification

## Common Development Commands

```bash
# Environment setup
pip install -r requirements.txt

# Data collection (when implemented)
python src/data/collectors/semantic_scholar_api.py --collect-papers --limit 10000
python src/data/collectors/dblp_scraper.py --venue "ICML,NeurIPS,ICLR" --years 2020-2023
python src/data/processors/temporal_splitter.py --split-date 2022-01-01

# Feature extraction
python src/features/text_features/specter_embeddings.py --input data/processed/papers.json
python src/features/network_features/citation_graph.py --build-graph --temporal
python src/features/social_features/altmetrics.py --collect-social-signals

# Model training
python src/models/transformers/specter_finetuned.py --train --early-prediction
python src/models/graph_models/graphsage_model.py --train --temporal-encoding
python src/models/fusion_models/adaptive_fusion.py --train --multi-modal

# Evaluation and bias analysis
python src/evaluation/bias_detection/temporal_bias.py --analyze --dataset data/processed/
python src/evaluation/fairness_metrics/demographic_parity.py --evaluate --models checkpoints/
python src/evaluation/evaluation_protocols/temporal_evaluation.py --test-early-prediction

# Prediction and explainability
python src/prediction/explainers/shap_explainer.py --explain --model checkpoints/best_model.pt
python src/prediction/interactive/prediction_api.py --serve --port 8000

# Testing
pytest tests/ -v
pytest tests/unit/ --cov=src/
pytest tests/integration/ --slow
```

## Technical Architecture

### Data Sources Integration
- **Semantic Scholar Academic Graph**: 200M+ papers with embeddings and citations
- **DBLP**: 6M+ CS publications with author networks and venue information
- **ArXiv**: 1.7M+ papers with full-text content
- **S2ORC**: 12M+ parsed papers with citation contexts
- **Altmetrics**: Social media mentions, news coverage, Wikipedia citations

### Model Architecture Components
- **Text Processing**: SPECTER2 + SciBERT with domain adaptation
- **Graph Processing**: GraphSAGE with temporal encoding
- **Multi-Modal Fusion**: Cross-modal attention with adaptive weighting
- **Temporal Modeling**: RNN-Transformer hybrid for citation dynamics
- **Bias Mitigation**: Adversarial training with fairness constraints

### Performance Targets
- **Citation Prediction MAE**: < 7.35 (beat current SOTA)
- **High-Impact Classification**: > 85% accuracy
- **Early Prediction**: Reliable predictions within 30-90 days
- **Multi-Modal Improvement**: > 6.4% AUC over single-modal methods

## Development Guidelines

### Code Quality
- Follow PEP 8 style guidelines with Black formatting
- Use type hints for all function signatures
- Write comprehensive docstrings for all modules and functions
- Maintain test coverage > 80% for core functionality

### Experiment Tracking
- Use MLflow or Weights & Biases for experiment logging
- Track all hyperparameters, metrics, and model artifacts
- Version control datasets and maintain data lineage
- Document all experimental decisions and results

### Reproducibility
- Set random seeds for all stochastic processes
- Use configuration files (YAML) for all hyperparameters
- Containerize environments with Docker
- Maintain detailed environment specifications

### Ethics and Bias
- Continuously monitor for systematic biases
- Implement fairness metrics in all evaluations
- Document potential ethical implications
- Follow responsible AI deployment practices

## Data Handling Best Practices

### API Rate Limiting
- Implement exponential backoff for API calls
- Cache API responses to minimize redundant requests
- Use batch processing for large-scale data collection
- Monitor API usage quotas and costs

### Data Quality
- Validate data integrity at each processing stage
- Handle missing data with appropriate strategies
- Detect and flag potential data quality issues
- Maintain data provenance and lineage tracking

### Privacy and Security
- Never store personal information unnecessarily
- Anonymize author data where possible
- Follow data protection regulations (GDPR compliance)
- Secure API keys and credentials using environment variables

## Model Development Workflow

### 1. Baseline Development
- Start with simple traditional ML models (SVM, Random Forest)
- Establish performance baselines on standard metrics
- Implement proper cross-validation with temporal awareness
- Document baseline performance for comparison

### 2. Advanced Model Development
- Implement transformer-based models (SPECTER, SciBERT)
- Develop graph neural networks for citation modeling
- Create multi-modal fusion architectures
- Experiment with hybrid approaches

### 3. Domain Specialization
- Adapt models for CS-specific characteristics
- Implement venue-aware and community-aware modeling
- Create subfield-specific prediction models
- Handle conference vs journal dynamics

### 4. Bias Mitigation
- Implement bias detection in training pipeline
- Apply fairness constraints during training
- Use adversarial debiasing techniques
- Validate fairness metrics across demographics

### 5. Explainability Integration
- Implement SHAP and LIME explainers
- Create attention visualization tools
- Develop causal discovery methods
- Build interactive explanation interfaces

## Deployment Considerations

### API Development
- Use FastAPI for REST API development
- Implement proper authentication and rate limiting
- Add comprehensive API documentation with OpenAPI
- Include health checks and monitoring endpoints

### Scalability
- Design for horizontal scaling with containerization
- Use batch processing for large-scale predictions
- Implement efficient caching strategies
- Monitor resource usage and performance metrics

### Monitoring and Maintenance
- Set up performance monitoring with Prometheus/Grafana
- Implement bias monitoring with automated alerts
- Create model drift detection systems
- Plan for regular model retraining and updates

## Troubleshooting Common Issues

### Memory Issues
- Use data generators for large datasets
- Implement gradient checkpointing for large models
- Consider model parallelism for transformer models
- Use efficient data structures (sparse matrices, compressed formats)

### Training Instability
- Implement gradient clipping for unstable training
- Use learning rate scheduling and warmup
- Monitor for exploding/vanishing gradients
- Add regularization techniques as needed

### Data Pipeline Issues
- Implement robust error handling in data collection
- Use data validation schemas (Pydantic models)
- Create data quality monitoring dashboards
- Plan for data source failures and recovery

This comprehensive guide provides the foundation for developing a state-of-the-art research paper virality prediction system with a focus on early prediction, multi-modal integration, CS-domain specialization, and ethical AI practices.