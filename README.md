# Early Prediction of Research Paper Virality in Computer Science

A comprehensive Final Year Project (FYP) focused on developing machine learning models that can predict research paper impact within 30-90 days of publication, addressing the critical gap in current methods that require 3+ years of citation data.

## 🎯 Project Objectives

- **Early Timeline Prediction**: Achieve reliable predictions within 30-90 days (vs current 3+ year requirement)
- **Multi-Modal Integration**: Combine text, network, temporal, and social signals for enhanced prediction
- **CS-Domain Specialization**: Build models specifically tailored for computer science publication patterns
- **Bias Mitigation**: Address systematic biases across temporal, venue, author, and geographic dimensions
- **Explainable AI**: Provide interpretable insights into virality mechanisms

## 📊 Key Performance Targets

| Metric | Target | Current SOTA |
|--------|--------|--------------|
| Citation Prediction MAE | < 7.35 | 7.352 |
| High-Impact Classification | > 85% | 71-85% |
| Prediction Timeline | 30-90 days | 3+ years |
| Multi-Modal Improvement | > 6.4% AUC | N/A |

## 🏗️ Project Structure

```
FYP/
├── 📁 src/                      # Source code modules
│   ├── data/                    # Data collection & processing
│   ├── features/                # Multi-modal feature engineering
│   ├── models/                  # ML/DL model implementations
│   ├── evaluation/              # Bias detection & fairness metrics
│   ├── prediction/              # Explainability & deployment
│   └── utils/                   # Utility functions
├── 📁 data/                     # Data storage (raw, processed, external)
├── 📁 experiments/              # Experiment tracking & results
├── 📁 notebooks/                # Jupyter notebooks for exploration
├── 📁 docs/                     # Documentation
├── 📁 tests/                    # Testing framework
├── 📄 PROJECT_BREAKDOWN.md      # Comprehensive implementation plan
├── 📄 CLAUDE.md                 # Development guidance
└── 📄 requirements.txt          # Python dependencies
```

## 🔬 Research Innovation Areas

### 1. **Temporal Constraint Innovation**
- **Problem**: Current methods need 3+ years of citation data
- **Solution**: Novel architectures optimized for sparse early signals
- **Impact**: Enable practical academic discovery systems

### 2. **Multi-Modal Fusion Architecture**
- **Problem**: Single-modality approaches miss cross-modal relationships
- **Solution**: Adaptive fusion with cross-modal attention mechanisms
- **Impact**: >6.4% AUC improvement over single-modal methods

### 3. **CS-Domain Specialization**
- **Problem**: General methods ignore CS-specific characteristics
- **Solution**: Conference dynamics modeling and subfield-specific patterns
- **Impact**: Specialized prediction for theory/systems/AI papers

### 4. **Bias-Aware Prediction**
- **Problem**: Systematic biases across multiple dimensions
- **Solution**: Adversarial debiasing with fairness constraints
- **Impact**: Ethical AI deployment in academic evaluation

## 📚 Data Sources

| Source | Coverage | Key Features |
|--------|----------|--------------|
| **Semantic Scholar** | 200M+ papers | SPECTER2 embeddings, 1.2B+ citations |
| **DBLP** | 6M+ CS papers | Author networks, venue information |
| **ArXiv** | 1.7M+ papers | Full-text content, early access |
| **S2ORC** | 12M+ papers | Parsed full-text, citation contexts |
| **Altmetrics** | 30-40% coverage | Social media, news, Wikipedia |

## 🤖 Technical Architecture

### Core Components
- **Text Processing**: SPECTER2 + SciBERT with domain adaptation
- **Graph Processing**: GraphSAGE with temporal encoding
- **Multi-Modal Fusion**: Cross-modal attention with adaptive weighting
- **Temporal Modeling**: RNN-Transformer hybrid for citation dynamics
- **Bias Mitigation**: Adversarial training with fairness constraints
- **Explainability**: SHAP + causal discovery for interpretable predictions

### Technology Stack
- **ML/DL**: PyTorch, Transformers, Scikit-learn, XGBoost
- **Graph Processing**: PyTorch Geometric, NetworkX, DGL
- **NLP**: SpaCy, NLTK, Sentence-Transformers
- **Experiment Tracking**: MLflow, Weights & Biases
- **Deployment**: FastAPI, Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 100GB+ storage space

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd FYP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models and data
python src/utils/setup_models.py
```

### Quick Start
```bash
# Data collection (example)
python src/data/collectors/semantic_scholar_api.py --collect-papers --limit 1000

# Feature extraction
python src/features/text_features/specter_embeddings.py --input data/processed/papers.json

# Model training
python src/models/fusion_models/adaptive_fusion.py --train --config configs/baseline.yaml

# Evaluation
python src/evaluation/evaluation_protocols/temporal_evaluation.py --model checkpoints/best_model.pt
```

## 📋 Development Phases

### Phase 1: Foundation (Weeks 1-3) ✅
- [x] Project structure setup
- [x] Data collection APIs
- [x] Preprocessing pipelines
- [x] Quality control frameworks

### Phase 2: Feature Engineering (Weeks 4-6)
- [ ] Multi-modal feature extraction
- [ ] Text embeddings (SPECTER2, SciBERT)
- [ ] Network analysis (citation graphs, author networks)
- [ ] Temporal features (publication timing, early signals)
- [ ] Social features (altmetrics, sentiment analysis)

### Phase 3: Model Development (Weeks 7-10)
- [ ] Baseline implementations
- [ ] Transformer-based models
- [ ] Graph neural networks
- [ ] Multi-modal fusion architectures
- [ ] CS-domain specialization

### Phase 4: Bias Mitigation (Weeks 11-12)
- [ ] Bias detection systems
- [ ] Fairness metrics implementation
- [ ] Adversarial debiasing
- [ ] Evaluation framework

### Phase 5: Explainability (Weeks 13-14)
- [ ] SHAP/LIME integration
- [ ] Causal discovery methods
- [ ] Interactive prediction systems
- [ ] Robustness testing

### Phase 6: Deployment (Weeks 15-16)
- [ ] Production API development
- [ ] Comprehensive validation
- [ ] Documentation and dissemination
- [ ] Open-source release

## 📈 Expected Outcomes

### Academic Contributions
- **Novel Architecture**: First system for 30-90 day paper virality prediction
- **Domain Specialization**: CS-specific modeling with subfield awareness
- **Bias Mitigation**: Comprehensive fairness evaluation framework
- **Explainable AI**: Interpretable virality mechanism identification

### Practical Impact
- **Academic Discovery**: Enable early identification of impactful research
- **Funding Decisions**: Inform research investment based on predicted impact
- **Career Development**: Help researchers understand virality factors
- **Conference Planning**: Assist in paper selection and session organization

## 📖 Documentation

- [`PROJECT_BREAKDOWN.md`](PROJECT_BREAKDOWN.md) - Comprehensive implementation plan with technical details
- [`CLAUDE.md`](CLAUDE.md) - Development guidance for future contributors
- [`Research.md`](Research.md) - Original research analysis and gap identification
- `docs/` - Additional technical documentation (to be created)

## 🤝 Contributing

This is an active research project. Contributions are welcome in the following areas:

- **Data Collection**: New data sources and API integrations
- **Feature Engineering**: Novel feature extraction methods
- **Model Architecture**: Advanced deep learning architectures
- **Bias Detection**: New fairness metrics and mitigation strategies
- **Evaluation**: Comprehensive benchmarking protocols

## 📄 License

[To be determined - likely Apache 2.0 or MIT for open source research]

## 📧 Contact

[Contact information to be added]

---

**Note**: This project is part of a Final Year Project (FYP) focused on advancing the state-of-the-art in computational scientometrics through novel machine learning approaches for early research impact prediction.