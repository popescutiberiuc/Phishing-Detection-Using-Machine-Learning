# phishing_detector/data_processing.py

import pandas as pd
import re
import nltk
import os
import kagglehub

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_and_merge_datasets_from_kagglehub():
    """Load only the combined phishing_email.csv dataset."""
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    print("Path to dataset files:", path)

    df = pd.read_csv(os.path.join(path, "phishing_email.csv"))
    df = df.rename(columns={"text_combined": "combined_text"})
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    print(f"Loaded {len(df)} samples from phishing_email.csv")
    return df[["combined_text", "label"]]


def clean_text(text):
    """Clean and normalize text content."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    return text

def simple_tokenize(text):
    """Simple tokenization function."""
    return text.split()

def preprocess_text_column(df, column="email"):
    """Apply cleaning and tokenization to a specified column."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def process(text):
        cleaned = clean_text(text)
        tokens = word_tokenize(cleaned)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)

    df[column] = df[column].apply(process)
    return df
