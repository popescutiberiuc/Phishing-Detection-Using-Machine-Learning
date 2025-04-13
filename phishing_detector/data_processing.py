# phishing_detector/data_processing.py

import pandas as pd
import re
import nltk
import os

# Download NLTK data directly - this is the key fix
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Import after download to ensure resources are available
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_and_merge_datasets(nazario_path, spamassassin_path):
    """Load and merge the phishing and normal email datasets."""
    # Check if files exist
    if not os.path.exists(nazario_path):
        raise FileNotFoundError(f"File not found: {nazario_path}")
    if not os.path.exists(spamassassin_path):
        raise FileNotFoundError(f"File not found: {spamassassin_path}")
    
    # Load datasets
    phishing_df = pd.read_csv(nazario_path)
    mixed_df = pd.read_csv(spamassassin_path)
    
    # Print dataset info
    print(f"Nazario dataset shape: {phishing_df.shape}")
    print(f"SpamAssassin dataset shape: {mixed_df.shape}")
    
    # Merge datasets
    combined_df = pd.concat([phishing_df, mixed_df], ignore_index=True)
    
    # Check for missing values
    combined_df.fillna('', inplace=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def clean_text(text):
    """Clean and normalize text content."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def simple_tokenize(text):
    """Simple tokenization function that doesn't rely on NLTK's sentence tokenizer."""
    return text.split()

def preprocess_text(text):
    """Apply full text preprocessing pipeline with simplified tokenization."""
    try:
        # Clean the text
        text = clean_text(text)
        
        # Simple tokenization as fallback
        try:
            tokens = word_tokenize(text)
        except:
            tokens = simple_tokenize(text)
        
        # Remove stopwords if available
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except:
            pass  # Skip stopword removal if there's an issue
        
        # Lemmatize if available
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except:
            pass  # Skip lemmatization if there's an issue
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        # Return cleaned text if processing fails
        return clean_text(text)

def extract_email_domain(email):
    """Extract domain from email address."""
    try:
        if not isinstance(email, str) or '@' not in email:
            return ""
        return email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return ""

def preprocess_dataset(df):
    """Apply preprocessing to the entire dataset."""
    print("Starting dataset preprocessing...")
    
    # Apply preprocessing in batches to avoid memory issues
    batch_size = 1000
    total_rows = len(df)
    
    # Initialize new columns
    df['clean_subject'] = ""
    df['clean_body'] = ""
    df['sender_domain'] = ""
    
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        print(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} (rows {i}-{end_idx-1})...")
        
        # Process subject
        df.loc[i:end_idx-1, 'clean_subject'] = df.loc[i:end_idx-1, 'subject'].astype(str).apply(clean_text)
        
        # Process body 
        df.loc[i:end_idx-1, 'clean_body'] = df.loc[i:end_idx-1, 'body'].astype(str).apply(clean_text)
        
        # Extract email domains
        df.loc[i:end_idx-1, 'sender_domain'] = df.loc[i:end_idx-1, 'sender'].astype(str).apply(extract_email_domain)
    
    # Create a combined text field for feature extraction
    df['combined_text'] = df['clean_subject'] + ' ' + df['clean_body']
    
    print("Preprocessing complete.")
    return df