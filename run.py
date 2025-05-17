import argparse
import os
import pandas as pd
from phishing_detector.pipeline import PhishingDetector
from phishing_detector.email_parser import parse_eml_file, load_labeled_eml_dataset
from phishing_detector.model_training import create_features_and_train
from sklearn.metrics import confusion_matrix, classification_report

def train_model():
    create_features_and_train()

def evaluate_real_dataset(model_type):
    model_path = f"models/phishing_detector_model_{model_type}.pkl"
    dataset_path = "real_eml_dataset"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Train model first with --train.")
        return

    print("✅ Loading real-world test dataset...")
    test_df = load_labeled_eml_dataset(dataset_path)

    X_real = test_df["combined_text"]
    y_real = test_df["label"]

    print(f"Real-world test set size: {len(y_real)} emails")

    detector = PhishingDetector(model_type=model_type)
    y_pred = []

    for i, text in enumerate(X_real):
        result = detector.process_email("", "", text)
        y_pred.append(int(result['is_phishing']))

    print("\n📋 Evaluation Report:")
    print(confusion_matrix(y_real, y_pred))
    print(classification_report(y_real, y_pred))

def analyze_folder(folder_path, model_type):
    model_path = f"models/phishing_detector_model_{model_type}.pkl"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Train model first with --train.")
        return

    print(f"📥 Scanning emails from folder: {folder_path}")
    email_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.lower().endswith('.eml')
    ]

    if not email_files:
        print("⚠️ No .eml files found in folder. Exiting.")
        return

    detector = PhishingDetector(model_type=model_type)
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
    parser.add_argument("--train", action="store_true", help="Train both models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on real_eml_dataset")
    parser.add_argument("--folder", type=str, help="Path to folder of .eml files to analyze")
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "rf"], help="Model type: lr or rf")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_real_dataset(args.model)
    elif args.folder:
        analyze_folder(args.folder, args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
