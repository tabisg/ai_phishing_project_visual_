import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import argparse

p = Path(__file__).parent

def train(dataset=None, out="phish_model.joblib"):
    if dataset is None:
        df = pd.read_csv(p / "sample_phishing.csv")
    else:
        df = pd.read_csv(dataset)
    X = df['url'].astype(str)
    y = df['label'].astype(int)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtrain, ytrain)
    pred = pipe.predict(Xtest)
    print("Accuracy:", accuracy_score(ytest, pred))
    print(classification_report(ytest, pred))
    joblib.dump(pipe, p / out)
    print(f"Model saved to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CSV dataset path (url,label)")
    ap.add_argument("--out", default="phish_model.joblib")
    args = ap.parse_args()
    train(args.data, args.out)
