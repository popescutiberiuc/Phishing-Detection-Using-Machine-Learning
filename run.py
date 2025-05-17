# run.py

import argparse
import os
import pandas as pd
import pickle
from phishing_detector.pipeline import PhishingDetector
from phishing_detector.email_parser import process_eml_directory, parse_eml_file, load_labeled_eml_dataset
from phishing_detector.model_training import create_features_and_train
from sklearn.metrics import confusion_matrix, classification_report

def train_model():
    create_features_and_train("models/phishing_detector_model.pkl")

def evaluate_real_dataset():
    model_path = "models/phishing_detector_model.pkl"
    dataset_path = "real_eml_dataset"

    if not os.path.exists(model_path):
        print("❌ Model file not found. Train model first with --train.")
        return

    print("✅ Loading real-world test dataset...")
    test_df = load_labeled_eml_dataset(dataset_path)

    X_real = test_df["combined_text"]
    y_real = test_df["label"]

    print(f"Real-world test set size: {len(y_real)} emails")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("\n📋 Evaluating on real-world dataset...")
    y_pred = model.predict(X_real)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_real, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_real, y_pred))

def analyze_folder(folder_path):
    model_path = "models/phishing_detector_model.pkl"

    if not os.path.exists(model_path):
        print("❌ Model file not found. Train model first with --train.")
        return

    print(f"📥 Scanning emails from folder: {folder_path}")
    email_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.eml'):
                email_files.append(os.path.join(root, file))

    if not email_files:
        print("⚠️ No .eml files found in folder. Exiting.")
        return

    detector = PhishingDetector(model_path)
    predictions = []

    for file_path in email_files:
        filename = os.path.basename(file_path)
        email_data = parse_eml_file(file_path)

        if email_data:
            sender = email_data['sender']
            subject = email_data['subject']
            body = email_data['body']

            result = detector.process_email(sender, subject, body)

            predictions.append({
                'filename': filename,
                'sender': sender,
                'subject': subject,
                'is_phishing': result['is_phishing'],
                'phishing_probability': result['phishing_probability'],
                'confidence': result['confidence']
            })

            print(f"📧 {filename}: {'Phishing' if result['is_phishing'] else 'Not Phishing'} "
                  f"(prob: {result['phishing_probability']:.2f}, confidence: {result['confidence']})")

    results_df = pd.DataFrame(predictions)
    try:
        results_df.to_csv("phishing_analysis_results.csv", index=False)
        print(f"\n✅ Results saved to phishing_analysis_results.csv")
    except PermissionError:
        print("\n❌ Cannot write results file. Is it open in Excel or another program?")

def main():
    parser = argparse.ArgumentParser(description="Phishing Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train model on Kaggle dataset")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on real_eml_dataset")
    parser.add_argument("--folder", type=str, help="Path to folder of .eml files to analyze")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_real_dataset()
    elif args.folder:
        analyze_folder(args.folder)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()