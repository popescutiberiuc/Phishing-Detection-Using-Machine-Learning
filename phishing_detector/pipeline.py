# phishing_detector/pipeline.py

import pickle
import pandas as pd
from .data_processing import preprocess_text, extract_email_domain
import re

class PhishingDetector:
    def __init__(self, model_path):
        """Initialize the detector with a trained model."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def extract_urls(self, text):
        """Check if text contains URLs."""
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        if url_pattern.search(text):
            return 1
        return 0
    
    def process_email(self, sender, subject, body):
        """Process a single email and return phishing probability."""
        # Extract features
        clean_subject = preprocess_text(subject)
        clean_body = preprocess_text(body)
        combined_text = clean_subject + ' ' + clean_body
        sender_domain = extract_email_domain(sender)
        contains_urls = self.extract_urls(body)
        
        # Create DataFrame for prediction
        email_df = pd.DataFrame({
            'combined_text': [combined_text],
            'sender_domain': [sender_domain],
            'urls': [contains_urls]
        })
        
        # Make prediction
        prediction = self.model.predict(email_df)[0]
        probability = self.model.predict_proba(email_df)[0][1]
        
        return {
            'is_phishing': bool(prediction),
            'phishing_probability': float(probability),
            'confidence': 'High' if abs(probability - 0.5) > 0.4 else 'Medium' if abs(probability - 0.5) > 0.2 else 'Low'
        }