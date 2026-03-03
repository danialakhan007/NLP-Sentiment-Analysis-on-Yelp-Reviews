import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Page config
st.set_page_config(
    page_title="Sentiment Analysis - Multi-Model Prediction",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E75B6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Sentiment Analysis System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Multi-Model Comparison | Yelp Review Classification</p>', unsafe_allow_html=True)

# Sidebar - Model Info
st.sidebar.title("Model Performance")
st.sidebar.markdown("### Trained on 100K Yelp Reviews")

model_metrics = {
    "DistilBERT": {"f1": 93.61, "acc": 93.68, "type": "Deep Learning"},
    "Logistic Regression": {"f1": 92.00, "acc": 92.00, "type": "Classical ML"},
    "Linear SVM": {"f1": 91.72, "acc": 91.71, "type": "Classical ML"},
    "XGBoost": {"f1": 89.20, "acc": 89.24, "type": "Classical ML"}
}

for model_name, metrics in model_metrics.items():
    st.sidebar.markdown(f"**{model_name}**")
    st.sidebar.markdown(f"F1: {metrics['f1']:.2f}% | Acc: {metrics['acc']:.2f}%")
    st.sidebar.markdown(f"*{metrics['type']}*")
    st.sidebar.markdown("---")

st.sidebar.markdown("### About")
st.sidebar.info(
    "This app demonstrates production-grade sentiment analysis "
    "using 4 different ML models trained on Yelp restaurant reviews."
)

# Cache model loading
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    try:
        # Load TF-IDF vectorizer
        models['tfidf'] = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load classical ML models
        models['lr'] = joblib.load('models/logistic_regression.pkl')
        models['svm'] = joblib.load('models/linear_svm.pkl')
        models['xgb'] = joblib.load('models/xgboost.pkl')
        
        # Load DistilBERT
        models['bert_tokenizer'] = DistilBertTokenizer.from_pretrained('models/distilbert_final')
        models['bert_model'] = DistilBertForSequenceClassification.from_pretrained('models/distilbert_final')
        models['bert_model'].eval()
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure all model files are in the 'models/' directory")
        return None

# Load models
with st.spinner("Loading models..."):
    models = load_models()

if models is None:
    st.stop()

st.success("All models loaded successfully!")

# Main interface
st.markdown("##Enter Review Text")

# Text input
review_text = st.text_area(
    "Type or paste a restaurant review below:",
    height=150,
    placeholder="Example: The food was amazing and the service was excellent! Highly recommend this place."
)

# Example reviews
st.markdown("### Try these examples:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Positive Example"):
        review_text = "This restaurant is absolutely amazing! The food is delicious, service is great, and the atmosphere is wonderful. Highly recommend!"
        st.rerun()

with col2:
    if st.button("Negative Example"):
        review_text = "Terrible experience. The food was cold, service was horrible, and we waited forever. Never coming back."
        st.rerun()

with col3:
    if st.button("Mixed Example"):
        review_text = "The food was good but the service was really slow. Nice atmosphere though."
        st.rerun()

# Predict button
if st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("Please enter some text to analyze")
    else:
        with st.spinner("Analyzing with all models..."):
            
            # Transform text for classical models
            text_tfidf = models['tfidf'].transform([review_text])
            
            # Predictions dictionary
            predictions = {}
            
            # Logistic Regression
            lr_pred = int(models['lr'].predict(text_tfidf)[0])  # Convert to int
            lr_proba = models['lr'].predict_proba(text_tfidf)[0]
            predictions['Logistic Regression'] = {
            'prediction': 'Positive' if lr_pred == 1 else 'Negative',
            'confidence': float(lr_proba[lr_pred]) * 100,
            'probabilities': {'Negative': float(lr_proba[0]), 'Positive': float(lr_proba[1])}
            }
            
            # Linear SVM
            svm_pred = models['svm'].predict(text_tfidf)[0]
            svm_decision = models['svm'].decision_function(text_tfidf)[0]
            # Convert decision function to probability-like score
            svm_confidence = (1 / (1 + np.exp(-svm_decision))) * 100 if svm_pred == 1 else (1 / (1 + np.exp(svm_decision))) * 100
            predictions['Linear SVM'] = {
                'prediction': 'Positive' if svm_pred == 1 else 'Negative',
                'confidence': svm_confidence,
                'probabilities': None  # SVM doesn't give probabilities directly
            }
            
            # XGBoost
            xgb_pred = int(models['xgb'].predict(text_tfidf)[0])  # Convert to int
            xgb_proba = models['xgb'].predict_proba(text_tfidf)[0]
            predictions['XGBoost'] = {
            'prediction': 'Positive' if xgb_pred == 1 else 'Negative',
            'confidence': float(xgb_proba[xgb_pred]) * 100,
            'probabilities': {'Negative': float(xgb_proba[0]), 'Positive': float(xgb_proba[1])}
            }

            # DistilBERT
            inputs = models['bert_tokenizer'](
                review_text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = models['bert_model'](**inputs)
                bert_proba = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                bert_pred = torch.argmax(bert_proba).item()
            
            predictions['DistilBERT'] = {
                'prediction': 'Positive' if bert_pred == 1 else 'Negative',
                'confidence': bert_proba[bert_pred].item() * 100,
                'probabilities': {'Negative': bert_proba[0].item(), 'Positive': bert_proba[1].item()}
            }
        
        # Display results
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        # Create columns for results
        cols = st.columns(4)
        
        for idx, (model_name, result) in enumerate(predictions.items()):
            with cols[idx]:
                sentiment = result['prediction']
                confidence = result['confidence']
                
                # Color-coded box
                if sentiment == 'Positive':
                    st.markdown(f'<div class="positive">', unsafe_allow_html=True)
                    st.markdown(f"### {model_name}")
                    st.markdown(f"**Positive**")
                    st.markdown(f"Confidence: **{confidence:.1f}%**")
                else:
                    st.markdown(f'<div class="negative">', unsafe_allow_html=True)
                    st.markdown(f"### {model_name}")
                    st.markdown(f"**Negative**")
                    st.markdown(f"Confidence: **{confidence:.1f}%**")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show probability distribution if available
                if result['probabilities']:
                    st.progress(result['probabilities']['Positive'])
                    st.caption(f"Pos: {result['probabilities']['Positive']:.2%} | Neg: {result['probabilities']['Negative']:.2%}")
        
        # Consensus
        st.markdown("---")
        st.markdown("## Model Consensus")
        
        positive_count = sum(1 for p in predictions.values() if p['prediction'] == 'Positive')
        negative_count = 4 - positive_count
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if positive_count > negative_count:
                st.success(f"**Consensus: POSITIVE** ({positive_count}/4 models agree)")
            elif negative_count > positive_count:
                st.error(f"**Consensus: NEGATIVE** ({negative_count}/4 models agree)")
            else:
                st.warning("**Split Decision** (2-2 tie)")
        
        # Model comparison chart
        st.markdown("### Model Confidence Comparison")
        
        chart_data = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Confidence': [p['confidence'] for p in predictions.values()],
            'Prediction': [p['prediction'] for p in predictions.values()]
        })
        
        st.bar_chart(chart_data.set_index('Model')['Confidence'])
        
        # SHAP Explanation (for Logistic Regression)
        st.markdown("---")
        st.markdown("## Explainability (Logistic Regression)")
        
        with st.expander("Show SHAP Feature Importance"):
            try:
                # Create SHAP explainer for LR
                explainer = shap.LinearExplainer(models['lr'], text_tfidf)
                shap_values = explainer.shap_values(text_tfidf)
                
                # Get feature names
                feature_names = models['tfidf'].get_feature_names_out()
                
                # Get top contributing features
                feature_importance = np.abs(shap_values[0])
                top_indices = np.argsort(feature_importance)[-10:][::-1]
                
                st.markdown("**Top 10 words influencing this prediction:**")
                
                for idx in top_indices:
                    feature = feature_names[idx]
                    importance = shap_values[0][idx]
                    
                    if importance > 0:
                        st.markdown(f"- **{feature}** → Positive (SHAP: +{importance:.3f})")
                    else:
                        st.markdown(f"- **{feature}** → Negative (SHAP: {importance:.3f})")
                
            except Exception as e:
                st.warning(f"SHAP analysis unavailable: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>ML-Intensive Sentiment Analysis System</strong></p>
    <p>Trained on 100,000 Yelp restaurant reviews | 4 models | 93.6% F1 score (DistilBERT)</p>
    <p>Built with Streamlit • PyTorch • scikit-learn • HuggingFace Transformers</p>
</div>
""", unsafe_allow_html=True)