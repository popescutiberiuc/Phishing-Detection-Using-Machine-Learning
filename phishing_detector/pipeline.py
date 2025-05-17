import pickle
import pandas as pd
import re
from .data_processing import clean_text

def extract_email_domain(email_address):
    parts = email_address.split("@")
    return parts[1] if len(parts) == 2 else ""

class PhishingDetector:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def extract_urls(self, text):
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        return 1 if url_pattern.search(text) else 0
    
    def process_email(self, sender, subject, body):
        clean_subject = clean_text(subject)
        clean_body = clean_text(body)
        combined_text = clean_subject + ' ' + clean_body
        sender_domain = extract_email_domain(sender)
        contains_urls = self.extract_urls(body)

        email_df = pd.DataFrame({
            'combined_text': [combined_text],
            'sender_domain': [sender_domain],
            'urls': [contains_urls]
        })

        # only pass combined_text to model pipeline
        X_text = email_df['combined_text']
        threshold = 0.6
        probability = self.model.predict_proba(X_text)[0][1]
        prediction = 1 if probability > threshold else 0

        return {
            'is_phishing': bool(prediction),
            'phishing_probability': float(probability),
            'confidence': 'High' if abs(probability - 0.5) > 0.4 else
                          'Medium' if abs(probability - 0.5) > 0.2 else
                          'Low'
        }