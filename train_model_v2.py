
import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 55, flush=True)
print("       PHISHSHIELD AI - FINAL MODEL TRAINING", flush=True)
print("=" * 55, flush=True)

# 1. Load dataset
print("\n[1/6] Loading dataset...", flush=True)

df = pd.read_csv("dataset/cleaned_urls.csv")
df = df.dropna(subset=["url", "type"])
df["url"] = df["url"].astype(str)

# Use stratified sample to reduce memory usage
print("Selecting balanced sample...", flush=True)

df, _ = train_test_split(
    df,
    train_size=100000,
    random_state=42,
    stratify=df["type"]
)

X = df["url"]
y = df["type"]

print("Total records:", len(df), flush=True)
print(y.value_counts(), flush=True)

# 2. Split data
print("\n[2/6] Splitting dataset...", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train), flush=True)
print("Testing:", len(X_test), flush=True)

# 3. TF-IDF feature extraction
print("\n[3/6] Extracting URL patterns...", flush=True)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 4),
    min_df=3,
    max_features=15000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Features:", X_train_tfidf.shape, flush=True)

# 4. Train model
print("\n[4/6] Training model...", flush=True)

model = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

start = time.time()
model.fit(X_train_tfidf, y_train)

print("Training completed in",
      round(time.time() - start, 2), "seconds", flush=True)

# 5. Evaluate model
print("\n[5/6] Evaluating model...", flush=True)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test, predictions, zero_division=0
)
matrix = confusion_matrix(y_test, predictions)

print("\nACCURACY:", round(accuracy * 100, 2), "%", flush=True)
print("\nCLASSIFICATION REPORT:\n", report, flush=True)
print("\nCONFUSION MATRIX:\n", matrix, flush=True)

# Save evaluation report
os.makedirs("reports", exist_ok=True)

with open("reports/model_v2_evaluation.txt", "w") as f:
    f.write("PHISHSHIELD AI - VERSION 2 EVALUATION\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

# 6. Save model
print("\n[6/6] Saving model...", flush=True)

os.makedirs("models", exist_ok=True)

joblib.dump(vectorizer, "models/url_vectorizer_v2.pkl")
joblib.dump(model, "models/url_classifier_v2.pkl")

# Test popular websites
print("\nTESTING SAMPLE WEBSITES", flush=True)

test_urls = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "http://secure-login-verify-account.example.com/login"
]

test_features = vectorizer.transform(test_urls)
test_predictions = model.predict(test_features)
test_probabilities = model.predict_proba(test_features)

for url, prediction, probability in zip(
    test_urls, test_predictions, test_probabilities
):
    print("\nURL:", url, flush=True)
    print("Prediction:", prediction, flush=True)
    print("Model confidence:",
          round(max(probability) * 100, 2), "%", flush=True)

print("\n" + "=" * 55, flush=True)
print("       TRAINING COMPLETED SUCCESSFULLY!", flush=True)
print("=" * 55, flush=True)

print("Model: models/url_classifier_v2.pkl", flush=True)
print("Vectorizer: models/url_vectorizer_v2.pkl", flush=True)
print("Report: reports/model_v2_evaluation.txt", flush=True)
