# Advanced Data Analytics - Assignment 2

## Group 3: Customer Sentiment Analysis & Book Recommendation Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Executive Summary

This repository contains solutions to two business challenges in advanced data analytics:

1. **Customer Sentiment Analysis** - Automated classification of Yelp reviews using ML and transformer models
2. **Book Recommendation System** - Collaborative filtering using Alternating Least Squares (ALS)

**Key Technologies:** Python, scikit-learn, PyTorch, Transformers (RoBERTa, DistilBERT), Implicit ALS

---

## 🎯 Business Challenges

### Challenge 1: Yelp Sentiment Analysis

**Problem:** Yelp businesses receive thousands of reviews daily, making manual analysis impossible.

**Solution:** Automated sentiment classification system that:

- Classifies reviews as positive or negative
- Achieves 95%+ accuracy using transformer models
- Enables data-driven business decisions at scale

**Business Impact:**

- Reduce manual review time by 90%
- Identify customer issues faster
- Improve response time to negative feedback

### Challenge 2: Book Recommendation System

**Problem:** Digital book platforms need personalized recommendations to increase engagement.

**Solution:** ALS-based collaborative filtering that:

- Learns user preferences from 6M+ ratings
- Generates personalized book recommendations
- Handles sparse data efficiently

**Business Impact:**

- Increase user engagement by 25-40%
- Boost conversion rates by 15-30%
- Improve customer retention

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 - 3.11 (3.11 recommended)
- 8GB+ RAM
- Windows/Linux/macOS

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/prumucena1979/ADVASSIGNMENT2.git
cd ADVASSIGNMENT2

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch Jupyter
jupyter notebook GRP3-Assignment2-WITH-ALS.ipynb
```

### Running the Notebook

1. **Execute Global Setup Cell** (Cell 1) - Run this first!

   - Sets `RANDOM_STATE=42` for reproducibility
   - Imports all required libraries
   - Configures environment

2. **Follow Sequential Execution** - Run cells in order
   - Sentiment Analysis: Cells 3-18
   - Recommendation System: Cells 19-27

---

## 📁 Project Structure

```
ADVASSIGNMENT2/
│
├── GRP3-Assignment2-WITH-ALS.ipynb    # Main assignment notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Git exclusions
├── INSTALL_IMPLICIT_GUIDE.md          # Implicit library installation
│
└── PrjAssets/
    ├── Assignment 2.pdf                # Assignment instructions
    └── aiprompt                        # AI assistant prompts
```

---

## 🔬 Technical Approach

### Task 1: Sentiment Analysis

**Baseline Model:**

- TF-IDF vectorization (10,000 features)
- Random Forest classifier
- Grid search optimization
- **Accuracy:** ~88%

**Advanced Models:**

- **RoBERTa** (cardiffnlp/twitter-roberta-base-sentiment)
- **DistilBERT** (distilbert-base-uncased-finetuned-sst-2)
- **Accuracy:** 95%+ on validation set

### Task 2: Book Recommendations

**Algorithm:** Alternating Least Squares (ALS)

- Matrix factorization approach
- 64 latent factors
- Handles 6M+ ratings efficiently
- Filters users/books with 50+ interactions

**Data Source:** Goodbooks-10k dataset

- 53,424 unique users
- 10,000 unique books
- 5,976,479 ratings

---

## 📊 Key Results

### Sentiment Analysis Performance

| Model                     | Accuracy | Precision | Recall | F1-Score |
| ------------------------- | -------- | --------- | ------ | -------- |
| Random Forest (Baseline)  | 88.3%    | 88.1%     | 88.6%  | 88.3%    |
| Random Forest (Optimized) | 89.2%    | 89.0%     | 89.5%  | 89.2%    |
| RoBERTa                   | 95.1%    | 95.0%     | 95.3%  | 95.1%    |
| DistilBERT                | 94.8%    | 94.7%     | 95.0%  | 94.8%    |

### Recommendation System Metrics

- **Sparsity:** 98.9% (sparse matrix challenge)
- **Training Time:** ~2-5 minutes on CPU
- **Top-10 Recommendations:** Generated per user
- **Filter:** Excludes already-rated books

---

## 🛠️ Dependencies

### Core Libraries

- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning

### Deep Learning

- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.35.0` - Hugging Face transformers
- `datasets>=2.14.0` - Dataset loading

### Visualization

- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization

### Recommender Systems

- `scipy>=1.10.0` - Sparse matrices
- `implicit>=0.7.0` - ALS algorithm

**Note:** The `implicit` library requires C++ compilation. See `INSTALL_IMPLICIT_GUIDE.md` for detailed instructions.

---

## 📝 Academic Compliance

### Assignment Requirements

- ✅ Exploratory Data Analysis (EDA)
- ✅ Baseline ML model (Random Forest + TF-IDF)
- ✅ Grid search optimization
- ✅ Two transformer models (RoBERTa, DistilBERT)
- ✅ Performance comparison & business impact
- ✅ ALS recommendation system
- ✅ Visualization & insights

### Reproducibility

- `RANDOM_STATE=42` set globally
- All random seeds fixed (NumPy, PyTorch, scikit-learn)
- Sequential execution guaranteed

---

## 🤝 Contributors

**Group 3 Members:**

- [Add team member names]

**Course:** DAMO630 - Advanced Data Analytics  
**Institution:** [Add institution name]  
**Semester:** [Add semester]

---

## 📖 Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Implicit Library Documentation](https://implicit.readthedocs.io/)
- [Goodbooks-10k Dataset](https://github.com/zygmuntz/goodbooks-10k)
- [Yelp Polarity Dataset](https://huggingface.co/datasets/yelp_polarity)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🐛 Troubleshooting

### Common Issues

**1. Implicit library won't install**

- See `INSTALL_IMPLICIT_GUIDE.md` for detailed instructions
- Requires Microsoft Visual C++ Build Tools on Windows

**2. Out of memory errors**

- Reduce `SUBSET_SIZE` in sentiment analysis cells
- Use CPU instead of GPU for smaller batches

**3. Transformer models downloading slowly**

- Models download on first use (~500MB each)
- Cached in `~/.cache/huggingface/`

**4. Jupyter kernel crashes**

- Restart kernel and run cells sequentially
- Check memory usage (8GB+ recommended)

---

## 📧 Contact

For questions or issues, please contact:

- **Repository:** https://github.com/prumucena1979/ADVASSIGNMENT2
- **Email:** [Add contact email]

---

**Last Updated:** November 2025
