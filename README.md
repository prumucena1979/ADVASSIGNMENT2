# ML Group Assignment #2 - Group 3

## Overview
This project addresses two machine learning business challenges:
1. **Customer Sentiment Analysis** on Yelp reviews
2. **Book Recommendation System** using collaborative filtering

## Business Challenge 1: Yelp Sentiment Analysis

### Objective
Develop automated sentiment analysis to understand customer opinions at scale from Yelp reviews, enabling data-driven decisions to improve service quality and competitiveness.

### Dataset
- **Source**: Yelp Polarity Dataset (from Hugging Face datasets library)
- **Size**: 560,000 training samples
- **Classes**: Binary (Negative=0, Positive=1)
- **Balance**: Perfectly balanced dataset (50% negative, 50% positive)

### Tasks Completed

#### Task I: Exploratory Data Analysis
- Analyzed class distribution and confirmed balanced dataset
- Extracted sample reviews from both sentiment categories
- Analyzed review length distribution (mean: ~710 characters)
- Compared length patterns between positive and negative reviews

#### Task II: Baseline Model Development
- **Model**: TF-IDF + Random Forest Classifier
- **Features**: 10,000 TF-IDF features
- **Initial Performance**: ~87% accuracy on validation set
- **Optimization**: Grid Search CV with hyperparameter tuning
  - Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
  - Best configuration identified and validated
- **Evaluation Metrics**: Precision, Recall, F1-Score, Confusion Matrix

#### Task III: Advanced Transformer Models
- **Models Implemented**:
  1. **RoBERTa-Large** (siebert/sentiment-roberta-large-english)
     - State-of-the-art performance
     - Pre-trained specifically for sentiment analysis
  2. **DistilBERT** (distilbert-base-uncased-finetuned-sst-2-english)
     - Efficient alternative
     - Faster inference with competitive accuracy

- **Comparison**: Performance benchmarking against optimized Random Forest baseline
- **Business Impact**: Quantified improvements in accuracy, reduction in false negatives (preventing overlooked unhappy customers), and better precision (reducing escalation costs)

## Business Challenge 2: Book Recommendation System

### Objective
Develop a recommendation system for a digital book platform to identify similar books and recommend books that readers are most likely to enjoy based on past ratings and collaborative patterns.

### Dataset
- **Source**: Goodbooks-10k Dataset
- **URL**: https://github.com/zygmuntz/goodbooks-10k
- **Size**: 6+ million ratings
- **Users**: 50,000+ users
- **Books**: 10,000 unique books
- **Rating Scale**: 1 (lowest) to 5 (highest)

### Tasks Completed

#### Task I: Exploratory Data Analysis
- Dataset summary: rows, columns, variable types
- Rating distribution visualization
- User activity analysis (filtered users with <50 ratings)
- Book popularity analysis (filtered books with <50 ratings)
- Distribution plots (histograms, bar charts)

#### Task II: Collaborative Filtering with ALS
- **Model**: Alternating Least Squares (ALS) from `implicit` library
- **Approach**: Matrix Factorization technique
- **Implementation**:
  - Converted dataset to user-item interaction matrix
  - Created sparse matrix representation (Item-User format)
  - Trained ALS model on filtered dataset
  
- **Hyperparameters**:
  - **Factors**: 64 (latent dimensions)
  - **Regularization**: 0.05 (prevent overfitting)
  - **Alpha**: 2.0 (confidence multiplier)
  - **Iterations**: 20 (optimization steps)

- **Output**: 
  - Learned user preference vectors
  - Learned book embedding vectors
  - Top-N recommendations for test users

## Technologies Used

### Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Deep Learning**: transformers (Hugging Face), torch
- **Recommender Systems**: implicit (ALS implementation)
- **Datasets**: datasets (Hugging Face)

### Models
- Random Forest Classifier (baseline)
- RoBERTa-Large (transformer)
- DistilBERT (transformer)
- Alternating Least Squares (collaborative filtering)

## Key Findings

### Sentiment Analysis
- Balanced dataset with equal positive/negative reviews
- Negative reviews tend to be slightly longer than positive ones
- Transformer models significantly outperform traditional ML approaches
- RoBERTa achieves state-of-the-art performance for sentiment classification

### Recommendation System
- ALS effectively learns user preferences and book similarities
- Matrix factorization captures latent patterns in user-book interactions
- Filtering inactive users and unpopular books improves recommendation quality
- System successfully generates personalized top-N recommendations

## How to Run

### Prerequisites
```bash
pip install datasets transformers torch scikit-learn matplotlib seaborn pandas numpy implicit scipy
```

### Execution
Open and run the Jupyter notebook:
```
GRP3-Assignment2.ipynb
```

Execute cells sequentially to reproduce all analyses and results.

## Project Structure
```
.
├── GRP3-Assignment2.ipynb    # Main analysis notebook
└── README.md                  # Project documentation
```

## Contributors
Group 3

## License
Educational project - Machine Learning Course Assignment

---
*Note: This project demonstrates practical applications of NLP sentiment analysis and collaborative filtering recommendation systems for real-world business challenges.*
