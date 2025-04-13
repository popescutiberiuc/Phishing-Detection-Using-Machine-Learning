# phishing_detector/model_training.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

def create_features_and_train(df, model_output_path):
    """Extract features and train the model."""
    # Separate features and target
    X = df[['combined_text', 'sender_domain', 'urls']]
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create feature extraction pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_features', TfidfVectorizer(max_features=5000), 'combined_text'),
            ('domain_features', TfidfVectorizer(max_features=500), 'sender_domain')
        ],
        remainder='passthrough'
    )
    
    # Create and train the pipeline
    pipeline = Pipeline([
        ('features', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline, (X_test, y_test)