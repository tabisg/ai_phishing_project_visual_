import streamlit as st
import pandas as pd
from pathlib import Path
import joblib, io
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from utils import extract_url_features
from train_and_save import train as train_model

st.set_page_config(page_title="Phishing URL Detector — Visual", page_icon="🛡️", layout="wide")
st.title("🛡️ Phishing URL Detector — Visual Interface")
st.markdown("Enter a URL to check, or upload a dataset to train the model. Visual metrics and charts will help explain results to non-technical users.")

p = Path(__file__).parent
model_path = p / "phish_model.joblib"

# Left column: single-url check
col1, col2 = st.columns([1,1])

with col1:
    st.header("🔗 Check a single URL")
    url = st.text_input("Enter URL", "https://example.com/login")
    if st.button("Check URL"):
        if not model_path.exists():
            st.info("Model not found. Training a small demo model...")
            train_model(None)
        model = joblib.load(model_path)
        pred = model.predict([url])[0]
        prob = model.predict_proba([url])[0]
        label = "Phishing" if pred==1 else "Legitimate"
        confidence = max(prob)
        if pred==1:
            st.error(f"⚠️ Likely PHISHING ({confidence*100:.1f}% confidence)")
        else:
            st.success(f"✅ Likely legitimate ({confidence*100:.1f}% confidence)")
        st.subheader("Why the model thinks so (simple features)")
        st.json(extract_url_features(url))

with col2:
    st.header("📁 Upload dataset to train (optional)")
    st.markdown("CSV format: two columns — `url`, `label` (0 = legit, 1 = phishing)")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    use_sample = st.checkbox("Use example dataset (recommended for quick demo)", value=True)
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Dataset loaded: {len(df)} rows")
            st.dataframe(df.head(10))
            if st.button("Train model on uploaded dataset"):
                tmp = p / "uploaded_dataset.csv"
                df.to_csv(tmp, index=False)
                with st.spinner("Training..."):
                    train_model(str(tmp))
                st.success("Model trained and saved (phish_model.joblib).")
                model_path = p / "phish_model.joblib"
        except Exception as e:
            st.error("Failed to read CSV. Ensure it has 'url' and 'label' columns. Error: " + str(e))
    else:
        if use_sample:
            df = pd.read_csv(p / "sample_phishing.csv")
            st.write("Using built-in demo dataset. Sample:")
            st.dataframe(df.head(10))

st.markdown("---")
st.header("📊 Model Evaluation (on demo test split)")
if st.button("Show evaluation on demo dataset"):
    df = pd.read_csv(p / "sample_phishing.csv")
    X = df['url'].astype(str)
    y = df['label'].astype(int)
    # train a fresh model to get test split metrics
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6))), ('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(Xtrain, ytrain)
    ypred = pipe.predict(Xtest)
    yprob = pipe.predict_proba(Xtest)[:,1]
    acc = accuracy_score(ytest, ypred)
    st.metric("Accuracy on demo test split", f"{acc*100:.2f}%")
    st.subheader("Classification report")
    st.text(classification_report(ytest, ypred))
    # Confusion matrix
    cm = confusion_matrix(ytest, ypred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix (rows=true, cols=pred)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    st.pyplot(fig, use_container_width=True)
    # ROC curve
    fpr, tpr, _ = roc_curve(ytest, yprob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.set_title(f"ROC curve (AUC = {roc_auc:.2f})")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    st.pyplot(fig2, use_container_width=True)

st.markdown("---")
st.info("Tips: Use a larger real dataset for better accuracy (Phishtank, Kaggle). The dataset upload allows you to show improved metrics in your report.")
