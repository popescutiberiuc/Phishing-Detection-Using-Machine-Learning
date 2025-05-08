#!/usr/bin/env python

import os
import argparse
import sys
import traceback
from phishing_detector.data_processing import load_and_merge_datasets_from_kagglehub
from phishing_detector.model_training import create_features_and_train
from phishing_detector.pipeline import PhishingDetector
from phishing_detector.email_parser import process_eml_directory, save_emails_to_csv
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict on a single email')
    parser.add_argument('--text', type=str, help='Combined text for prediction')
    parser.add_argument('--process-eml', type=str, help='Process .eml files from directory')
    parser.add_argument('--output-csv', type=str, default='processed_emails.csv', help='Output CSV file for processed emails')
    parser.add_argument('--analyze-eml-dir', type=str, help='Analyze all .eml files in directory for phishing')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'phishing_detector_model.pkl')

    try:
        if args.train:
            print("Training model with dataset from Kaggle...")
            create_features_and_train(model_output_path=model_path)
            print(f"Model trained and saved to {model_path}")

        if args.process_eml:
            if not os.path.isdir(args.process_eml):
                print(f"Error: {args.process_eml} is not a valid directory")
                return

            print(f"Processing .eml files from {args.process_eml}...")
            emails_df = process_eml_directory(args.process_eml)
            save_emails_to_csv(emails_df, args.output_csv)

        if args.analyze_eml_dir:
            if not os.path.exists(model_path):
                print("Model not found. Please train the model first.")
                return

            if not os.path.isdir(args.analyze_eml_dir):
                print(f"Error: {args.analyze_eml_dir} is not a valid directory")
                return

            print(f"Analyzing .eml files from {args.analyze_eml_dir}...")
            emails_df = process_eml_directory(args.analyze_eml_dir)

            if emails_df.empty:
                print("No emails were processed.")
                return

            detector = PhishingDetector(model_path)

            for idx, row in emails_df.iterrows():
                text = row['combined_text']
                prediction = detector.model.predict([text])[0]
                probability = detector.model.predict_proba([text])[0][1]
                confidence = 'High' if abs(probability - 0.5) > 0.4 else 'Medium' if abs(probability - 0.5) > 0.2 else 'Low'

                emails_df.at[idx, 'is_phishing'] = prediction
                emails_df.at[idx, 'phishing_probability'] = probability
                emails_df.at[idx, 'confidence'] = confidence

            output_path = "phishing_analysis_results.csv"
            emails_df.to_csv(output_path, index=False)
            print(f"Analysis complete. Results saved to {output_path}")

            phishing_count = emails_df['is_phishing'].sum()
            print(f"\nSummary: {phishing_count} phishing emails detected out of {len(emails_df)}")

        if args.predict:
            if not os.path.exists(model_path):
                print("Model not found. Please train the model first.")
                return

            if not args.text:
                print("Please provide combined text using --text")
                return

            detector = PhishingDetector(model_path)
            prediction = detector.model.predict([args.text])[0]
            probability = detector.model.predict_proba([args.text])[0][1]
            confidence = 'High' if abs(probability - 0.5) > 0.4 else 'Medium' if abs(probability - 0.5) > 0.2 else 'Low'

            print("\nPhishing Detection Results:")
            print(f"Verdict: {'PHISHING' if prediction else 'LEGITIMATE'}")
            print(f"Confidence: {confidence}")
            print(f"Phishing Probability: {probability:.2f}")

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
