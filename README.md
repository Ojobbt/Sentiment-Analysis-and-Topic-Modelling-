# Sentiment-Analysis-and-Topic-Modelling-
# Sentiment Analysis on Amazon Product Reviews

## Project Overview

This project performs **binary sentiment classification** on Amazon product reviews using both:

1. **Traditional Machine Learning models (TF-IDF + classifiers)**
2. **Transformer-based Deep Learning models (DistilBERT)**

The objective is to compare classical NLP approaches with modern transformer-based methods in terms of:
- Accuracy
- Interpretability
- Computational complexity
- Model performance
## Dataset
The dataset used is:
**amazon_cells_labelled.txt**
- Format: Tab-separated file
- Columns:
  - `text` → Review text
  - `label` → Sentiment label  
    - `1` = Positive  
    - `0` = Negative  
### Example:
The dataset is cleaned and preprocessed before modeling.

## Project Structure

## Workflow

### 1. Data Loading & Inspection

- Load dataset with pandas
- Check for missing values
- Inspect class distribution
- Visualize label counts

### 2. Text Preprocessing

Text cleaning includes:

- Lowercasing
- Removing punctuation
- Removing digits
- Tokenization
- Stopword removal
- POS tagging
- Lemmatization

This step improves model generalization and reduces noise.

### 3. Exploratory Data Analysis (EDA)

Visualizations include:

- Label distribution
- Raw vs cleaned text length distribution
- Most frequent words
- Top TF-IDF features

These help understand:

- Class balance
- Text complexity
- Important vocabulary
- Feature distribution

### 4. Baseline Models (TF-IDF + ML)

The following classifiers were trained:

- Logistic Regression
- Linear SVM
- Naive Bayes
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

Pipeline structure: TF-IDF Vectorizer → Classifier

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A comparison bar chart visualizes performance across models.

### 5. Transformer Model (DistilBERT)

A transformer-based model was fine-tuned: distilbert-base-uncased
Process:

- Stratified train/test split
- Tokenization using Hugging Face tokenizer
- Fine-tuning with `Trainer`
- Accuracy metric computation
- Confusion matrix visualization
- Training & evaluation loss curves

Advantages of transformer approach:
- Context-aware embeddings
- Better semantic understanding
- Typically higher accuracy on complex text

## Model Comparison

| Approach | Pros | Cons |
|----------|------|------|
| TF-IDF + ML | Fast, interpretable, lightweight | Limited contextual understanding |
| DistilBERT | Strong contextual modeling | Computationally heavier |

In small datasets, classical ML often performs competitively.
Transformers show stronger potential as dataset size increases.

## Installation Requirements

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
pip install transformers datasets evaluate accelerate

## How to Run

1. Place dataset file in the project directory.
2. Open the notebook.
3. Run cells sequentially.
4. Review:
- EDA visualizations
- Model comparison results
- Transformer performance
## Reproducibility

- Random seed fixed (SEED = 42)
- Stratified train/test split
- Deterministic training where possible
## Key Learnings
1. Proper text preprocessing significantly improves ML performance.

2. TF-IDF remains strong for small datasets.

3. Transformers require more computation but capture deeper semantics.

4. Evaluation metrics beyond accuracy are essential for balanced interpretation.




