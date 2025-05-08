# phishing_detector/email_parser.py

import os
import email
import re
import pandas as pd
from email.header import decode_header
from email.utils import parseaddr

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

            # Plain text
            if content_type == "text/plain":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode(charset, errors='replace')
                except Exception as e:
                    print(f"Error decoding plain text part: {e}")

            # HTML (stripped of tags)
            elif content_type == "text/html":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_text = payload.decode(charset, errors='replace')
                        body += re.sub(r'<[^>]*>', ' ', html_text)
                except Exception as e:
                    print(f"Error decoding HTML part: {e}")
    else:
        # Not multipart
        content_type = msg.get_content_type()
        try:
            charset = msg.get_content_charset() or 'utf-8'
            payload = msg.get_payload(decode=True)
            if payload:
                text = payload.decode(charset, errors='replace')
                if content_type == "text/html":
                    text = re.sub(r'<[^>]*>', ' ', text)
                body += text
        except Exception as e:
            print(f"Error decoding non-multipart email: {e}")

    return body

def parse_eml_file(file_path):
    """Parse an .eml file and return combined_text suitable for prediction."""
    try:
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)

        sender_name, sender_email = parseaddr(msg.get('From', ''))
        receiver_name, receiver_email = parseaddr(msg.get('To', ''))
        subject = decode_email_header(msg.get('Subject', ''))
        date = msg.get('Date', '')
        body = get_email_body(msg)

        # Create combined text in the same style as the training dataset
        combined_text = f"From: {sender_email}\nTo: {receiver_email}\nDate: {date}\nSubject: {subject}\n\n{body}"

        return {
            'combined_text': combined_text
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

    if emails_data:
        return pd.DataFrame(emails_data)
    else:
        return pd.DataFrame(columns=["combined_text"])

def save_emails_to_csv(df, output_path):
    """Save processed emails to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} emails to {output_path}")
