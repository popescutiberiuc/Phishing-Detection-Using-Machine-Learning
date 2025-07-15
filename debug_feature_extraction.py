
from phishing_detector.email_parser import load_labeled_eml_dataset, parse_eml_file
from phishing_detector.feature_extraction import extract_features_from_email
import os

sample_path = "real_eml_dataset/ham"

sample_file = None
for file in os.listdir(sample_path):
    if os.path.isfile(os.path.join(sample_path, file)):
        sample_file = os.path.join(sample_path, file)
        break

if not sample_file:
    print(" No .eml file found in real_eml_dataset/ham/")
else:
    print(f" Loading sample email: {sample_file}\n")

    # Parse original email
    parsed = parse_eml_file(sample_file)
    print("=== Combined Text ===")
    print(parsed["combined_text"][:1000])  

    features = extract_features_from_email(parsed)
    print("\n=== Extracted Features ===")
    for key, value in features.items():
        print(f"{key}: {value}")