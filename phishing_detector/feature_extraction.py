
import re
from phishing_detector.utils import clean_text
from phishing_detector.pipeline import extract_email_domain

url_pattern = re.compile(r"http[s]?://|www\.")

def extract_features_from_email(email_data: dict):
    """Extract structured features from parsed email."""
    sender = email_data.get("sender", "")
    subject = email_data.get("subject", "")
    body = email_data.get("body", "")

    sender_domain = extract_email_domain(sender)
    contains_urls = 1 if url_pattern.search(body) else 0

    return {
        "sender_domain": sender_domain,
        "subject": clean_text(subject),
        "body": clean_text(body),
        "contains_urls": contains_urls
    }