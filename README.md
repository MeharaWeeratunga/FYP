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
```

## 📋 Development Status

### Phase 1: Foundation ✅
- [x] Project structure setup
- [x] Documentation creation
- [x] Requirements specification

### Upcoming Phases
- [ ] Data collection and preprocessing
- [ ] Multi-modal feature engineering
- [ ] Model development and training
- [ ] Bias detection and mitigation
- [ ] Explainability implementation
- [ ] Production deployment

## 📖 Documentation

- [`PROJECT_BREAKDOWN.md`](PROJECT_BREAKDOWN.md) - Comprehensive implementation plan
- [`CLAUDE.md`](CLAUDE.md) - Development guidance
- [`Research.md`](Research.md) - Original research analysis

## 🎯 Key Innovation Areas

- **Timeline Innovation**: 30-90 day predictions vs current 3+ year requirement
- **Multi-Modal Fusion**: Adaptive attention mechanisms across data types
- **CS-Domain Specialization**: Conference dynamics and subfield modeling
- **Bias Mitigation**: Comprehensive fairness evaluation framework
- **Explainable AI**: Causal discovery and interpretable predictions

---

**Final Year Project**: Early Prediction of Research Paper Virality in Computer Science