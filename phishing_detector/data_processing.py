import pandas as pd
import re
import nltk
import os
import kagglehub
from phishing_detector.email_parser import load_labeled_eml_dataset

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_and_merge_datasets_from_kagglehub():
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    print("Path to dataset files:", path)
    df = pd.read_csv(os.path.join(path, "phishing_email.csv"))
    df = df.rename(columns={"text_combined": "combined_text"})
    df["combined_text"] = df["combined_text"].fillna("").astype(str)
    print(f"Loaded {len(df)} samples from phishing_email.csv")
    return df[["combined_text", "label"]]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_tokenize(text):
    return text.split()

def preprocess_text_column(df, column="email"):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    def process(text):
        cleaned = clean_text(text)
        tokens = word_tokenize(cleaned)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)
    df[column] = df[column].apply(process).apply(clean_text)
    return df

def load_and_merge_datasets_from_kagglehub():
    # Load Kaggle dataset
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
    kaggle_df = pd.read_csv(os.path.join(path, "phishing_email.csv"))
    kaggle_df = kaggle_df.rename(columns={"text_combined": "combined_text"})
    kaggle_df["combined_text"] = kaggle_df["combined_text"].fillna("").astype(str)
    kaggle_df = kaggle_df[["combined_text", "label"]]

    # ✅ NEW: Load real-world dataset
    #real_eml_path = "real_eml_dataset"   # or whatever folder you choose
    #real_df = load_labeled_eml_dataset(real_eml_path)

    # Combine datasets
    #df = pd.concat([kaggle_df, real_df], ignore_index=True)
    #print(f"Kaggle samples: {len(kaggle_df)}, Real-world samples: {len(real_df)}, Total: {len(df)}")
    return kaggle_df