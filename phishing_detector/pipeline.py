import pickle
import pandas as pd
import re
from phishing_detector.data_processing import clean_text

def extract_email_domain(email_address):
    parts = email_address.split("@")
    return parts[1] if len(parts) == 2 else ""

class PhishingDetector:
    def __init__(self, model_type="lr"):
        model_path = f"models/phishing_detector_model.pkl_{model_type}.pkl"
        with open(model_path, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)

    def extract_urls(self, text):
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        return 1 if url_pattern.search(text) else 0

    def process_email(self, sender, subject, body):
        clean_subject = clean_text(subject)
        clean_body = clean_text(body)
        combined_text = clean_subject + ' ' + clean_body

        X_text = pd.Series([combined_text])
        X_vec = self.vectorizer.transform(X_text)

        probability = self.model.predict_proba(X_vec)[0][1]
        threshold = 0.6
        prediction = 1 if probability > threshold else 0

        return {
            'is_phishing': bool(prediction),
            'phishing_probability': float(probability),
            'confidence': 'High' if abs(probability - threshold) > 0.4 else
                          'Medium' if abs(probability - threshold) > 0.2 else
                          'Low'
        }