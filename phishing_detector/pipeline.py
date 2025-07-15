import pickle
import pandas as pd
import re
from phishing_detector.data_processing import clean_text

def extract_email_domain(email_address):
    
    parts = email_address.split("@")
    return parts[1] if len(parts) == 2 else ""

class PhishingDetector:
    def __init__(self, model_type="lr"):
        model_path = f"models/phishing_detector_model_{model_type}.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        
        if isinstance(data, tuple) and len(data) == 2:
            self.vectorizer, self.model = data
            self.subj_vectorizer = None
            self.body_vectorizer = None
        elif isinstance(data, tuple) and len(data) == 3:
            self.subj_vectorizer, self.body_vectorizer, self.model = data
            self.vectorizer = None
        else:
            raise ValueError(f"Unexpected model artifact format: {model_path}")

    def extract_urls(self, text):
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        return 1 if url_pattern.search(text) else 0

    def process_email(self, sender, subject, body):
        
        clean_subject = clean_text(subject)
        clean_body = clean_text(body)

        
        if self.vectorizer is not None:
            
            combined = clean_subject + ' ' + clean_body
            X = pd.Series([combined])
            X_vec = self.vectorizer.transform(X)
            prob = self.model.predict_proba(X_vec)[0][1]
        else:
            
            subj = pd.Series([clean_subject])
            body_series = pd.Series([clean_body])
            X_subj = self.subj_vectorizer.transform(subj) if self.subj_vectorizer else None
            X_body = self.body_vectorizer.transform(body_series)
            from scipy.sparse import hstack, csr_matrix
            if X_subj is not None and X_subj.shape[1] > 0:
                X_vec = hstack([X_subj, X_body])
            else:
                X_vec = X_body
            
            expected = self.model.n_features_in_
            actual = X_vec.shape[1]
            if actual < expected:
                pad = csr_matrix((1, expected - actual))
                X_vec = hstack([X_vec, pad])
            prob = self.model.predict_proba(X_vec)[0][1]

        # Prediction threshold
        threshold = 0.6
        pred = 1 if prob > threshold else 0
        conf = 'High' if abs(prob - threshold) > 0.4 else 'Medium' if abs(prob - threshold) > 0.2 else 'Low'

        return {
            'is_phishing': bool(pred),
            'phishing_probability': float(prob),
            'confidence': conf
        }
