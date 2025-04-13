# phishing_detector/email_parser.py

import os
import email
import re
import pandas as pd
from email.header import decode_header
from email.utils import parseaddr
import urllib.parse

def decode_email_header(header):
    """Decode email header to readable text."""
    if header is None:
        return ""
    
    decoded_parts = []
    for part, encoding in decode_header(header):
        if isinstance(part, bytes):
            try:
                if encoding:
                    decoded_parts.append(part.decode(encoding))
                else:
                    decoded_parts.append(part.decode('utf-8', errors='replace'))
            except Exception:
                decoded_parts.append(part.decode('utf-8', errors='replace'))
        else:
            decoded_parts.append(str(part))
    
    return ' '.join(decoded_parts)

def get_email_body(msg):
    """Extract the plain text body from an email message."""
    body = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
            
            # Get plain text content
            if content_type == "text/plain":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(charset, errors='replace')
                except Exception as e:
                    print(f"Error decoding part: {e}")
            
            # Extract text from HTML content if needed
            elif content_type == "text/html":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_text = payload.decode(charset, errors='replace')
                        # Simple HTML tag removal (you might want to use a more robust solution)
                        body += re.sub(r'<[^>]*>', ' ', html_text)
                except Exception as e:
                    print(f"Error decoding HTML part: {e}")
    else:
        # Not multipart - get the content directly
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                charset = msg.get_content_charset() or 'utf-8'
                payload = msg.get_payload(decode=True)
                if payload:
                    body += payload.decode(charset, errors='replace')
            except Exception as e:
                print(f"Error decoding message: {e}")
        elif content_type == "text/html":
            try:
                charset = msg.get_content_charset() or 'utf-8'
                payload = msg.get_payload(decode=True)
                if payload:
                    html_text = payload.decode(charset, errors='replace')
                    body += re.sub(r'<[^>]*>', ' ', html_text)
            except Exception as e:
                print(f"Error decoding HTML message: {e}")
    
    return body

def extract_urls_from_text(text):
    """Extract URLs from text content."""
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    urls = re.findall(url_pattern, text)
    return urls

def parse_eml_file(file_path):
    """Parse an .eml file and extract relevant information."""
    try:
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)
        
        # Extract sender and receiver
        from_header = msg.get('From', '')
        sender_name, sender_email = parseaddr(from_header)
        
        to_header = msg.get('To', '')
        receiver_name, receiver_email = parseaddr(to_header)
        
        # Extract date
        date = msg.get('Date', '')
        
        # Extract subject
        subject = decode_email_header(msg.get('Subject', ''))
        
        # Extract body
        body = get_email_body(msg)
        
        # Extract URLs from body
        urls = extract_urls_from_text(body)
        has_urls = 1 if urls else 0
        
        return {
            'sender': sender_email,
            'receiver': receiver_email,
            'date': date,
            'subject': subject,
            'body': body,
            'urls': has_urls,
            'label': None  # To be filled by the detector
        }
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def process_eml_directory(directory_path):
    """Process all .eml files in a directory and return a DataFrame."""
    emails_data = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.eml'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                
                email_data = parse_eml_file(file_path)
                if email_data:
                    emails_data.append(email_data)
    
    # Create DataFrame
    if emails_data:
        df = pd.DataFrame(emails_data)
        return df
    else:
        return pd.DataFrame(columns=['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label'])

def save_emails_to_csv(df, output_path):
    """Save the processed emails to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} emails to {output_path}")