# phishing_detector/model_training.py

import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from .data_processing import load_and_merge_datasets_from_kagglehub

def create_features_and_train(model_output_path):
    """Load data, train a model, and save it."""
    # Load preprocessed dataset (combined_text + label)
    df = load_and_merge_datasets_from_kagglehub()

    # Split features and labels
    X = df["combined_text"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline, (X_test, y_test)
