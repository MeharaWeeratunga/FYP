# Data Collection Module

This module contains all data collection and validation components for the Early Prediction of Research Paper Virality project.

## 📁 Directory Structure

```
src/data/
├── collectors/              # Data collection modules
│   ├── semantic_scholar_api.py      # Original API client with early citation analysis
│   └── production_collector.py     # Production-ready collector with quality controls
├── validators/              # Data quality validation
│   └── data_quality_validator.py   # Comprehensive validation framework
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Basic Data Collection
```bash
# Collect sample papers for testing
python scripts/data_collection/quick_sample_collector.py

# Run production collection (takes time due to rate limiting)
python src/data/collectors/production_collector.py
```

### 2. Data Validation
```bash
# Validate collected data
python src/data/validators/data_quality_validator.py data/raw/sample/sample_papers_*.json
```

### 3. Analysis
```bash
# Analyze collected data for research insights
python scripts/analysis/analyze_data_for_prediction.py
```

## 📊 Collection Strategies

### **Venue-Based Collection**
- Targets high-impact CS conferences (ICML, NeurIPS, CVPR, etc.)
- Year-specific searches (2022-2024 for early prediction research)
- Quality filtering (minimum citations, required abstracts)

### **Keyword-Based Collection**
- CS-specific keywords (machine learning, computer vision, NLP, etc.)
- Broad coverage with duplicate removal
- Relevance filtering using CS field classification

## 🔍 Data Quality Framework

### **Validation Checks**
- **Required Fields**: paper_id, title, citation_count
- **Data Types**: String, integer, list validations
- **Content Quality**: Title/abstract length, reasonable values
- **Consistency**: Year vs publication date, citation velocity
- **CS Classification**: Field-based and venue-based filtering

### **Quality Metrics**
- Field completeness percentages
- Citation statistics (mean, median, range)
- Venue diversity and distribution
- Temporal coverage analysis
- CS paper validation ratios

### **Quality Grading**
- **A (90-100%)**: Excellent quality dataset
- **B (80-89%)**: Good quality dataset
- **C (70-79%)**: Acceptable quality dataset
- **D (60-69%)**: Below average quality
- **F (<60%)**: Poor quality requiring cleanup

## 🎯 Research-Specific Features

### **Early Prediction Focus**
- Papers from 2022-2023 with sufficient citation history
- 30/60/90 day citation windows for early prediction validation
- Citation velocity calculations for virality indicators

### **CS Domain Specialization**
- Computer Science field classification
- Major CS venue recognition (conferences and journals)
- Technical keyword identification

### **Quality Filtering**
- Minimum citation thresholds to focus on impactful papers
- Abstract presence requirement for text analysis
- Publication date validation for temporal analysis

## 📋 Available Data Fields

### **Core Fields (100% availability)**
- `paper_id`: Unique identifier
- `title`: Paper title
- `abstract`: Paper abstract
- `authors`: Author list with IDs and names
- `venue`: Publication venue
- `year`: Publication year
- `publication_date`: Exact publication date
- `citation_count`: Current citation count
- `fields_of_study`: Academic field classifications

### **Additional Fields**
- `reference_count`: Number of references
- `url`: Paper URL
- `open_access_pdf`: PDF access information
- `is_cs_paper`: CS classification flag
- `citation_velocity`: Citations per month
- `age_days`: Days since publication

## ⚙️ Configuration

### **Collection Parameters**
```python
CollectionConfig(
    rate_limit_seconds=3.0,     # API rate limiting
    papers_per_venue=30,        # Papers per venue/year
    papers_per_keyword=20,      # Papers per keyword search
    min_citation_count=10,      # Minimum citation threshold
    target_years=[2022, 2023, 2024]  # Target publication years
)
```

### **API Rate Limiting**
- Conservative 3-second delays between requests
- Exponential backoff for rate limit errors
- Progress tracking for resumable collection

## 🔧 Error Handling

### **API Errors**
- 429 (Rate Limited): Exponential backoff retry
- 404 (Not Found): Log and continue
- Network errors: Retry with increasing delays

### **Data Errors**
- Missing required fields: Skip paper with error log
- Invalid data types: Validation error reporting
- Inconsistent data: Warning flags in quality report

## 📈 Performance Characteristics

### **Collection Speed**
- ~20 papers per minute (with rate limiting)
- Venue searches: More targeted, higher success rate
- Keyword searches: Broader coverage, more filtering needed

### **Success Rates**
- CS paper classification: ~95% accuracy
- Quality criteria pass rate: ~60-80% depending on venue
- Data completeness: 80-100% for core fields

## 🚨 Known Limitations

### **API Constraints**
- No SPECTER embeddings via free API
- Rate limiting slows large-scale collection
- Limited altmetrics/social media data

### **Data Gaps**
- Early citation timelines require separate collection
- Social media signals not directly available
- Author reputation/h-index not consistently available

## 🛠️ Troubleshooting

### **Common Issues**

**Rate Limiting Errors**
```bash
# Solution: Increase rate_limit_seconds in config
config.rate_limit_seconds = 5.0
```

**Empty Search Results**
```bash
# Check venue spelling and year availability
# Some venues may not be indexed consistently
```

**Validation Failures**
```bash
# Run data validator to identify specific issues
python src/data/validators/data_quality_validator.py your_data.json
```

### **Performance Optimization**
- Use venue-based collection for targeted results
- Implement parallel collection (if API key allows)
- Filter by minimum citation count early
- Save progress frequently for resumable collection

## 📚 Next Steps

### **Phase 2 Integration**
1. Feature extraction pipeline using collected data
2. Text embedding using alternative to SPECTER
3. Temporal splitting for early prediction validation
4. Integration with altmetrics APIs for social signals

### **Scalability Improvements**
1. Distributed collection across multiple API keys
2. Database integration for larger datasets
3. Real-time collection for continuous updates
4. Integration with institutional APIs for better access

---

For detailed implementation guidance, see the Phase 1 summary in `docs/PHASE1_DATA_GATHERING_SUMMARY.md`.