import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
from .data_processing import load_and_merge_datasets_from_kagglehub
from sklearn.metrics import confusion_matrix, classification_report

def create_features_and_train(model_output_path):
    df = load_and_merge_datasets_from_kagglehub()
    df['sender_domain'] = ""  # add dummy column for consistency
    df['urls'] = 0            # add dummy column for consistency

    X = df[['combined_text', 'sender_domain', 'urls']]
    y = df["label"]

    # Combine text + structured features into plain text for old TF-IDF pipeline
    X_combined = X['combined_text']

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)),
        ('classifier', LogisticRegression(
            C=0.5,
            solver='liblinear',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    with open(model_output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {model_output_path}")
