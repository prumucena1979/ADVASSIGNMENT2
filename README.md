# Advanced Machine Learning - Assignment 2: Group 3

## 📋 Overview

This assignment implements two comprehensive machine learning solutions for real-world business challenges:

1. **Customer Sentiment Analysis** - Analyzing Yelp reviews using NLP and transformer models
2. **Book Recommendation System** - Collaborative filtering using Alternating Least Squares (ALS)

---

## 🎯 Business Challenge 1: Yelp Sentiment Analysis

### Business Context

**Problem Statement**: Yelp receives millions of customer reviews, making it impossible for business managers to manually analyze customer sentiment. This project develops an automated sentiment analysis pipeline to understand customer opinions at scale and enable data-driven business decisions.

**Business Value**:

- Identify service quality issues faster
- Reduce customer churn through proactive response
- Optimize resource allocation based on sentiment trends
- Improve competitive positioning through customer insights

### Dataset

| Attribute            | Details                              |
| -------------------- | ------------------------------------ |
| **Source**           | Yelp Polarity Dataset (Hugging Face) |
| **Training Samples** | 560,000 reviews                      |
| **Test Samples**     | 38,000 reviews                       |
| **Classes**          | Binary (0=Negative, 1=Positive)      |
| **Balance**          | Perfectly balanced (50/50 split)     |
| **Features**         | Raw text reviews                     |

### Implementation Details

#### 📊 Task I: Exploratory Data Analysis

- **Class Distribution Analysis**: Confirmed balanced dataset with equal positive/negative reviews
- **Sample Review Extraction**: Displayed representative examples from both categories
- **Text Length Analysis**:
  - Mean review length: ~710 characters
  - Distribution visualization using histograms and KDE plots
  - Comparison of length patterns between sentiment classes
- **Key Insight**: Negative reviews tend to be slightly longer, suggesting customers provide more detail when dissatisfied

#### 🤖 Task II: Baseline Model Development

**Model Architecture**: TF-IDF + Random Forest Classifier

**Feature Engineering**:

- TF-IDF vectorization with 10,000 features
- Captures word importance across the corpus
- Handles sparse text representation efficiently

**Initial Performance** (Validation Set):

- Accuracy: ~87%
- Balanced performance across both classes

**Hyperparameter Optimization**:

- **Method**: Grid Search with 3-fold Cross-Validation
- **Search Space**:
  - `n_estimators`: [100, 500]
  - `max_depth`: [None, 20, 50]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [2, 4]
  - `max_features`: ["sqrt", "log2"]
- **Optimization Metric**: F1-Score
- **Result**: Best configuration identified and validated

**Evaluation**:

- Comprehensive classification report (Precision, Recall, F1-Score)
- Confusion matrix visualization
- Support metrics for both classes

#### 🚀 Task III: Advanced Transformer Models

**Model 1: Twitter-RoBERTa-Base**

- **Model ID**: `cardiffnlp/twitter-roberta-base-sentiment`
- **Architecture**: RoBERTa (Robustly Optimized BERT)
- **Advantages**:
  - Pre-trained on 58M tweets
  - Optimized for social media text and short reviews
  - Superior performance on informal language
- **Output**: 3-class (negative, neutral, positive) mapped to binary

**Model 2: DistilBERT**

- **Model ID**: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Architecture**: Distilled BERT (40% smaller, 60% faster)
- **Advantages**:
  - Fine-tuned on Stanford Sentiment Treebank (SST-2)
  - Efficient inference for production deployment
  - Maintains 97% of BERT's performance with lower cost
- **Output**: Binary sentiment classification

**Model Comparison**:

- Performance benchmarking against Random Forest baseline
- Metrics comparison: Accuracy, Precision, Recall, F1-Score
- Inference time analysis

**Business Impact Analysis**:

- **Improved Accuracy**: Better overall sentiment detection
- **Reduced False Negatives**: Fewer missed unhappy customers → proactive retention
- **Higher Precision**: Less noise in alerts → reduced escalation costs
- **ROI Justification**: Cost-benefit analysis of transformer deployment

---

## 📚 Business Challenge 2: Book Recommendation System

### Business Context

**Problem Statement**: Digital book platforms (similar to Goodreads/Amazon Kindle) need to recommend books that readers will enjoy based on historical ratings and collaborative patterns. Poor recommendations lead to reduced engagement and lost revenue.

**Business Value**:

- Increase user engagement and session time
- Drive book discovery and sales conversion
- Improve customer lifetime value
- Reduce churn through personalized experiences

### Dataset

| Attribute         | Details                                     |
| ----------------- | ------------------------------------------- |
| **Source**        | Goodbooks-10k (GitHub)                      |
| **Total Ratings** | 6+ million                                  |
| **Unique Users**  | 53,424                                      |
| **Unique Books**  | 10,000                                      |
| **Rating Scale**  | 1 (lowest) to 5 (highest)                   |
| **Sparsity**      | ~98.8% (typical for recommendation systems) |
| **Data Format**   | user_id, book_id, rating                    |

### Implementation Details

#### 📊 Task I: Exploratory Data Analysis

**Dataset Summary**:

- Rows: 5,976,479 ratings
- Columns: 3 (user_id, book_id, rating)
- Data types: All numerical

**Rating Distribution**:

- Modal rating: 4 stars (most common)
- Right-skewed distribution (users tend to rate books they enjoyed)
- Full distribution: 1★ to 5★ with counts and percentages

**Visualizations Created**:

1. **Rating Distribution Bar Chart**: Shows count of each rating level
2. **Rating Distribution Pie Chart**: Percentage breakdown
3. **User Activity Histogram**: Number of ratings per user (log scale)
4. **Book Popularity Histogram**: Number of ratings per book (log scale)

**Data Quality Measures**:

- **User Filtering**: Retained only users with ≥50 ratings (active users)
- **Book Filtering**: Retained only books with ≥50 ratings (popular books)
- **Rationale**: Improves recommendation quality by focusing on reliable patterns
- **Filtered Dataset**: ~1.5M ratings, ~3,000 users, ~2,000 books

#### 🧠 Task II: Collaborative Filtering with ALS

**Algorithm**: Alternating Least Squares (ALS) Matrix Factorization

**Mathematical Foundation**:

```
R ≈ U × V^T

Where:
- R: User-Item rating matrix (sparse)
- U: User factor matrix (users × latent_factors)
- V: Item factor matrix (items × latent_factors)
```

**Model Implementation**:

- **Library**: `implicit` (optimized C++ implementation)
- **Matrix Format**: CSR (Compressed Sparse Row) for memory efficiency
- **Training Matrix**: Item-User format (books × users)
- **Prediction Matrix**: User-Item format (users × books)

**Hyperparameter Configuration**:

| Parameter        | Value | Purpose                                      |
| ---------------- | ----- | -------------------------------------------- |
| `factors`        | 64    | Number of latent dimensions (expressiveness) |
| `regularization` | 0.05  | L2 penalty to prevent overfitting            |
| `iterations`     | 20    | Number of ALS optimization cycles            |
| `alpha`          | 2.0   | Confidence multiplier for implicit feedback  |
| `random_state`   | 42    | Reproducibility seed                         |
| `use_gpu`        | False | CPU-based training                           |

**Training Process**:

1. Matrix preparation and index mapping
2. ALS optimization with progress tracking
3. Factor matrix extraction (user and item embeddings)
4. Recommendation generation with filtering

**Output**:

- **User Factors**: Shape (n_users, 64) - User preference embeddings
- **Item Factors**: Shape (n_books, 64) - Book characteristic embeddings
- **Recommendations**: Top-N books per user with confidence scores

**Recommendation Features**:

- ✅ Filters already-rated books
- ✅ Includes confidence scores
- ✅ Displays book metadata (titles)
- ✅ Shows user's historical preferences for context

---

## 🛠️ Technologies & Tools

### Core Libraries

| Category                | Libraries            | Version |
| ----------------------- | -------------------- | ------- |
| **Data Processing**     | pandas, numpy        | Latest  |
| **Sparse Matrices**     | scipy.sparse         | Latest  |
| **Visualization**       | matplotlib, seaborn  | Latest  |
| **ML Framework**        | scikit-learn         | 1.7+    |
| **Deep Learning**       | transformers, torch  | Latest  |
| **NLP**                 | tokenizers, datasets | Latest  |
| **Recommender Systems** | implicit             | 0.7+    |

### Model Architecture Summary

| Task                 | Model                | Type                 | Parameters    |
| -------------------- | -------------------- | -------------------- | ------------- |
| Baseline Sentiment   | Random Forest        | Traditional ML       | ~10K features |
| Advanced Sentiment 1 | Twitter-RoBERTa-Base | Transformer          | 125M          |
| Advanced Sentiment 2 | DistilBERT           | Transformer          | 66M           |
| Recommendations      | ALS                  | Matrix Factorization | 64 factors    |

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher (3.11 recommended)
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8 GB minimum (16 GB recommended for transformer models)
- **Disk Space**: ~5 GB for datasets and models

### Installation

#### Step 1: Clone Repository

```bash
git clone <repository-url>
cd ADVASSIGNMENT2
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**⚠️ Important Note for Windows Users:**

The `implicit` library requires Microsoft C++ Build Tools. See [`INSTALL_IMPLICIT_GUIDE.md`](INSTALL_IMPLICIT_GUIDE.md) for detailed installation instructions.

**Quick Install Alternative (Conda)**:

```bash
conda install -c conda-forge implicit
```

### Running the Notebook

1. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

2. **Open Notebook**:

   - Navigate to `GRP3-Assignment2-WITH-ALS.ipynb`

3. **Execute Cells**:
   - Run cells sequentially (Cell → Run All)
   - Allow time for model downloads on first run
   - Transformer models will download ~500MB of pre-trained weights

### Expected Runtime

| Section                   | Time (Approx.)           |
| ------------------------- | ------------------------ |
| EDA (Sentiment)           | 2-3 minutes              |
| Baseline Model Training   | 5-10 minutes             |
| Transformer Model Loading | 1-2 minutes (first time) |
| Transformer Inference     | 3-5 minutes              |
| EDA (Recommendations)     | 2-3 minutes              |
| ALS Model Training        | 5-10 minutes             |
| **Total**                 | **20-35 minutes**        |

---

## 📁 Project Structure

```
ADVASSIGNMENT2/
│
├── GRP3-Assignment2-WITH-ALS.ipynb  # Main analysis notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── INSTALL_IMPLICIT_GUIDE.md        # Implicit library installation guide
│
├── PrjAssets/                       # Supporting materials
│   └── aiprompt                     # AI assistance documentation
│
└── .gitignore                       # Git ignore rules
```

---

## 🔍 Key Results & Findings

### Sentiment Analysis

**Dataset Characteristics**:

- ✅ Perfectly balanced classes (no class imbalance issues)
- ✅ Sufficient sample size for robust training
- ✅ Negative reviews are typically more detailed

**Model Performance**:

1. **Random Forest (Optimized)**: ~87-90% accuracy
2. **Twitter-RoBERTa**: ~92-95% accuracy (significant improvement)
3. **DistilBERT**: ~91-93% accuracy (good efficiency/accuracy trade-off)

**Business Insights**:

- Transformer models justify higher deployment costs through improved accuracy
- False negative reduction enables proactive customer retention
- DistilBERT offers best balance for production deployment

### Recommendation System

**System Characteristics**:

- ✅ Handles extreme sparsity (98.8% missing ratings)
- ✅ Learns latent user preferences and book characteristics
- ✅ Generates personalized recommendations with confidence scores

**Model Performance**:

- Successfully learned 64-dimensional embeddings
- Captures genre, style, and complexity patterns
- Recommendations align with user's historical preferences

**Business Insights**:

- Filtering improves quality by focusing on active users
- ALS scales well to millions of ratings
- Matrix factorization captures subtle preference patterns

---

## 📚 References & Resources

### Datasets

- [Yelp Polarity Dataset](https://huggingface.co/datasets/yelp_polarity)
- [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)

### Models

- [Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

### Libraries

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Implicit Library](https://implicit.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)

---

## 👥 Contributors

**Group 3** - Advanced Machine Learning Course

---

## 📄 License

Educational project for academic purposes only.

---

## 🆘 Troubleshooting

### Common Issues

**Issue**: `implicit` installation fails

- **Solution**: See [`INSTALL_IMPLICIT_GUIDE.md`](INSTALL_IMPLICIT_GUIDE.md) for detailed instructions

**Issue**: CUDA/GPU errors with transformers

- **Solution**: Models default to CPU. Set `device = "cpu"` explicitly if needed

**Issue**: Out of memory errors

- **Solution**: Reduce batch size or use smaller dataset subset

**Issue**: Slow transformer inference

- **Solution**: Use DistilBERT or reduce validation set size

---

## 📧 Contact & Support

For questions or issues related to this assignment, please contact the course instructor or teaching assistants.

---

_Last Updated: November 24, 2025_
