import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from phishing_detector.data_processing import load_and_merge_datasets_from_kagglehub

def create_features_and_train(model_output_path_base="models/phishing_detector_model"):
    print("✅ Loading Kaggle dataset...")
    df = load_and_merge_datasets_from_kagglehub()
    X = df["combined_text"]
    y = df["label"]

    print("📊 Label distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🔹 Train Logistic Regression
    lr = LogisticRegression(
        C=0.5,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr.fit(X_train_vec, y_train)
    print("\n📋 Logistic Regression Evaluation:")
    print(classification_report(y_test, lr.predict(X_test_vec)))

    with open(f"{model_output_path_base}_lr.pkl", "wb") as f:
        pickle.dump((vectorizer, lr), f)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🔹 Train Random Forest
    rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
    )
    rf.fit(X_train_vec, y_train)
    print("\n📋 Random Forest Evaluation:")
    print(classification_report(y_test, rf.predict(X_test_vec)))

    with open(f"{model_output_path_base}_rf.pkl", "wb") as f:
        pickle.dump((vectorizer, rf), f)

    print("\n✅ Both models saved!")

if __name__ == "__main__":
    create_features_and_train()
