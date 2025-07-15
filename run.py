import argparse
import os
import pandas as pd
from phishing_detector.pipeline import PhishingDetector
from phishing_detector.email_parser import parse_eml_file, load_labeled_eml_dataset
from phishing_detector.model_training import create_features_and_train
from sklearn.metrics import confusion_matrix, classification_report

# Ensemble weights
LR_WEIGHT = 0.65  
RF_WEIGHT = 0.35  

def train_model():
    create_features_and_train()


def evaluate_real_dataset(model_type):
    dataset_path = "real_eml_dataset"

    # model file check
    if model_type in ("lr", "rf"):
        model_path = f"models/phishing_detector_model_{model_type}.pkl"
        if not os.path.exists(model_path):
            print(f" Model file not found at {model_path}. Train model first with --train.")
            return
    else:  
        missing = []
        for m in ("lr", "rf"):
            p = f"models/phishing_detector_model_{m}.pkl"
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            print(" Missing model files for ensemble:", ", ".join(missing))
            print("   Train with --train to generate phishing_detector_model_lr.pkl and _rf.pkl")
            return

    print(" Loading real-world test dataset...")
    test_df = load_labeled_eml_dataset(dataset_path)

    X_real = test_df["combined_text"]
    y_real = test_df["label"]

    # single‐model evaluation
    if model_type in ("lr", "rf"):
        detector = PhishingDetector(model_type=model_type)
        y_pred = [
            int(detector.process_email("", "", txt)["is_phishing"])
            for txt in X_real
        ]

    # ensemble evaluation
    else:
        lr_det = PhishingDetector(model_type="lr")
        rf_det = PhishingDetector(model_type="rf")
        y_pred = []
        for txt in X_real:
            lr_p = lr_det.process_email("", "", txt)["phishing_probability"]
            rf_p = rf_det.process_email("", "", txt)["phishing_probability"]
            prob = LR_WEIGHT * lr_p + RF_WEIGHT * rf_p
            y_pred.append(int(prob > 0.5))

    print("\n Evaluation Report:")
    print(confusion_matrix(y_real, y_pred))
    print(classification_report(y_real, y_pred))



def analyze_folder(folder_path, model_type):
    print(f" Scanning emails from folder: {folder_path}")
    email_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.lower().endswith('.eml')
    ]

    if not email_files:
        print(" No .eml files found in folder. Exiting.")
        return

    # load detectors
    if model_type in ("lr", "rf"):
        detector = PhishingDetector(model_type=model_type)
        lr_det = rf_det = None
    else:
        lr_det = PhishingDetector(model_type="lr")
        rf_det = PhishingDetector(model_type="rf")
        detector = None

    predictions = []
    for file_path in email_files:
        filename = os.path.basename(file_path)
        email_data = parse_eml_file(file_path)
        if not email_data:
            continue
        sender = email_data['sender']
        subject = email_data['subject']
        body = email_data['body']

        if model_type in ("lr", "rf"):
            result = detector.process_email(sender, subject, body)
            pred = result['is_phishing']
            prob = result['phishing_probability']
            conf = result.get('confidence')
        else:
            # ensemble
            lr_res = lr_det.process_email(sender, subject, body)
            rf_res = rf_det.process_email(sender, subject, body)
            # weighted ensemble probability
            prob = (LR_WEIGHT * lr_res['phishing_probability'] + RF_WEIGHT * rf_res['phishing_probability']) / (LR_WEIGHT + RF_WEIGHT)
            pred = bool(prob > 0.7)
            conf = None

        predictions.append({
            'filename': filename,
            'sender': sender,
            'subject': subject,
            'is_phishing': pred,
            'phishing_probability': prob,
            'confidence': conf
        })

        print(f" {filename}: {'Phishing' if pred else 'Not Phishing'} "
              f"(prob: {prob:.2f}{', '+conf if conf else ''})")

    results_df = pd.DataFrame(predictions)
    try:
        results_df.to_csv("phishing_analysis_results.csv", index=False)
        print(f"\n Results saved to phishing_analysis_results.csv")
    except PermissionError:
        print("\n Cannot write results file. Is it open in Excel or another program?")


def main():
    parser = argparse.ArgumentParser(description="Phishing Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train both models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on real_eml_dataset")
    parser.add_argument("--folder", type=str, help="Path to folder of .eml files to analyze")
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "rf", "ens"], help="Model type: lr, rf, or ens")
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
