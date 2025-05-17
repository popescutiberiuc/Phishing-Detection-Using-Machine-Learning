# phishing_detector/utils.py

import re

def clean_text(text):
    """Standardizes text for model training and inference."""
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'\d+', ' ', text)      # remove numbers
    text = re.sub(r'[^a-z\s]', ' ', text) # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text