# 🎭 Sentiment Analysis on Yelp Reviews

**Production-grade NLP pipeline achieving 93.61% F1 score through systematic comparison of classical and deep learning models**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io/)

---

## 📊 Project Overview

Comprehensive sentiment analysis system for Yelp restaurant reviews featuring:

- **6 ML Models**: VADER, TextBlob, Logistic Regression, SVM, XGBoost, DistilBERT
- **4 Feature Methods**: TF-IDF, Word2Vec, Universal Sentence Encoder, BERT embeddings  
- **SHAP Explainability**: Transparent model interpretability
- **Interactive Demo**: Streamlit web application

### 🎯 Results

| Model | F1 Score | Type |
|-------|----------|------|
| **DistilBERT** | **93.61%** 🥇 | Deep Learning |
| Logistic Regression | 92.00% 🥈 | Classical ML |
| Linear SVM | 91.72% 🥉 | Classical ML |
| XGBoost | 89.20% | Classical ML |
| TextBlob | 61.41% | Baseline |
| VADER | 60.59% | Baseline |

**Achievement:** +33% improvement over baselines

---

## 📁 Repository Structure

```
├── 📂 Datasets/           # Sample data (10 rows for demo)
├── 📂 Notebooks/          # Complete ML pipeline
│   ├── 01_EDA_deep_dive.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_competition.ipynb
│   └── 04_shap_explainability.ipynb
├── 📂 Visuals/           # Plots & visualizations
├── 📄 app.py             # Streamlit application
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/danialakhan007/NLP-Sentiment-Analysis-on-Yelp-Reviews.git
cd NLP-Sentiment-Analysis-on-Yelp-Reviews
pip install -r requirements.txt
```

### Download Full Dataset

```python
from datasets import load_dataset
dataset = load_dataset("yelp_polarity")
```

### Run Streamlit App

```bash
streamlit run app.py
```

**Note:** Trained models hosted separately (1.2GB total)

---

## 🔬 Technical Approach

### 1. EDA (Notebook 01)
- Analyzed 100K stratified reviews
- Finding: Negative reviews 30% longer (152 vs 115 words)
- Vocabulary analysis revealed distinct sentiment patterns

### 2. Feature Engineering (Notebook 02)
| Method | Dimensions | Type |
|--------|------------|------|
| TF-IDF | 5,000 | Sparse |
| Word2Vec | 300 | Dense |
| USE | 512 | Dense |
| BERT | 384 | Dense |

### 3. Model Training (Notebook 03)
- **Baselines:** VADER, TextBlob (~61% F1)
- **Classical ML:** LR, SVM, XGBoost (89-92% F1)
- **Deep Learning:** DistilBERT fine-tuned (93.61% F1)

Training config: 3 epochs, batch 16, lr 2e-5, T4 GPU

### 4. Explainability (Notebook 04)
SHAP analysis revealed top predictive words:
- Positive: great (0.33), delicious (0.22), good (0.19)
- Negative: horrible (0.12), terrible (0.10)

---

## 📈 Key Insights

✅ Linear models (92% F1) nearly match deep learning (93.6%)  
✅ TF-IDF features capture sentiment effectively  
✅ DistilBERT handles rare words better than classical ML  
✅ SHAP confirms models learn meaningful patterns  

---

## 🛠️ Tech Stack

**Core:** Python 3.12, PyTorch 2.10, Transformers 4.35  
**ML:** scikit-learn, TextBlob, VADER, Logistic Regression, Linear SVM,  XGBoost, DistilBERT, SHAP  
**Deployment:** Streamlit, Hugging Face Spaces  

---

## 🎯 Model Downloads

Trained models available at:
- [Hugging Face Hub](#) *(coming soon)*

---

## 🔮 Future Work

- [ ] Multi-class (5-star) classification
- [ ] Aspect-based sentiment analysis  
- [ ] Model ensemble (LR + DistilBERT)
- [ ] Real-time API deployment

---

## 📧 Contact

**Danial Khan**  
📧 danialadnanseven@outlook.com  
💼 [LinkedIn](https://www.linkedin.com/in/danialadnankhan/)  
🐙 [GitHub](https://github.com/danialakhan007)

---

**⭐ Star this repo if helpful!**
