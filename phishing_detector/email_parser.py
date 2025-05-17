import os
import email
import re
import pandas as pd
from email.header import decode_header
from email.utils import parseaddr
from .utils import clean_text

def decode_email_header(header):
    if header is None:
        return ""
    decoded_parts = []
    for part, encoding in decode_header(header):
        if isinstance(part, bytes):
            try:
                decoded_parts.append(part.decode(encoding if encoding else 'utf-8'))
            except Exception:
                decoded_parts.append(part.decode('utf-8', errors='replace'))
        else:
            decoded_parts.append(str(part))
    return ' '.join(decoded_parts)

def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if "attachment" in content_disposition:
                continue
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
            elif content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode(part.get_content_charset() or 'utf-8', errors='replace')
                    body += re.sub(r'<[^>]*>', ' ', html)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text = payload.decode(msg.get_content_charset() or 'utf-8', errors='replace')
            if msg.get_content_type() == "text/html":
                text = re.sub(r'<[^>]*>', ' ', text)
            body += text
    return body

def parse_eml_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)
        sender_name, sender_email = parseaddr(msg.get('From', ''))
        receiver_name, receiver_email = parseaddr(msg.get('To', ''))
        subject = decode_email_header(msg.get('Subject', ''))
        date = msg.get('Date', '')
        body = get_email_body(msg)

        # clean subject + body for model use
        combined_clean = clean_text(subject) + ' ' + clean_text(body)

        return {
            'combined_text': combined_clean,
            'sender': sender_email,
            'subject': subject,
            'body': body
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def process_eml_directory(directory_path):
    emails_data = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.eml'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                email_data = parse_eml_file(file_path)
                if email_data:
                    emails_data.append(email_data)
    return pd.DataFrame(emails_data) if emails_data else pd.DataFrame(columns=["combined_text", "sender", "subject", "body"])

def save_emails_to_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} emails to {output_path}")

def load_labeled_eml_dataset(folder_path):
    """Load manually labeled eml dataset from folder structure."""
    data = []
    for label_folder, label in [('phishing', 1), ('ham', 0)]:
        path = os.path.join(folder_path, label_folder)
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            if file.lower().endswith('.eml'):
                file_path = os.path.join(path, file)
                email_data = parse_eml_file(file_path)
                if email_data:
                    data.append({
                        'combined_text': email_data['combined_text'],
                        'label': label
                    })
    return pd.DataFrame(data)