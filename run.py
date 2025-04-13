#!/usr/bin/env python
# run.py

import os
import argparse
import sys
import traceback
from phishing_detector.data_processing import load_and_merge_datasets, preprocess_dataset
from phishing_detector.model_training import create_features_and_train
from phishing_detector.pipeline import PhishingDetector
from phishing_detector.email_parser import process_eml_directory, save_emails_to_csv

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detection System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict on a new email')
    parser.add_argument('--sender', type=str, help='Email sender for prediction')
    parser.add_argument('--subject', type=str, help='Email subject for prediction')
    parser.add_argument('--body', type=str, help='Email body for prediction')
    parser.add_argument('--process-eml', type=str, help='Process .eml files from directory')
    parser.add_argument('--output-csv', type=str, default='processed_emails.csv', 
                        help='Output CSV file for processed emails')
    parser.add_argument('--analyze-eml-dir', type=str, 
                        help='Analyze all .eml files in directory for phishing')
    
    args = parser.parse_args()
    
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nazario_path = os.path.join(base_dir, 'data', 'Nazario.csv')
    spamassassin_path = os.path.join(base_dir, 'data', 'SpamAssasin.csv')
    model_path = os.path.join(base_dir, 'models', 'phishing_detector_model.pkl')
    
    try:
        if args.train:
            print("Loading and merging datasets...")
            combined_df = load_and_merge_datasets(nazario_path, spamassassin_path)
            
            print("Preprocessing data...")
            processed_df = preprocess_dataset(combined_df)
            
            print("Training model...")
            pipeline, test_data = create_features_and_train(processed_df, model_path)
            
            print(f"Model trained and saved to {model_path}")
        
        if args.process_eml:
            if not os.path.isdir(args.process_eml):
                print(f"Error: {args.process_eml} is not a valid directory")
                return
            
            print(f"Processing .eml files from {args.process_eml}...")
            emails_df = process_eml_directory(args.process_eml)
            
            output_path = args.output_csv
            save_emails_to_csv(emails_df, output_path)
        
        if args.analyze_eml_dir:
            if not os.path.exists(model_path):
                print("Model not found. Please train the model first.")
                return
                
            if not os.path.isdir(args.analyze_eml_dir):
                print(f"Error: {args.analyze_eml_dir} is not a valid directory")
                return
            
            print(f"Processing and analyzing .eml files from {args.analyze_eml_dir}...")
            emails_df = process_eml_directory(args.analyze_eml_dir)
            
            if emails_df.empty:
                print("No emails were processed.")
                return
                
            # Load the model
            detector = PhishingDetector(model_path)
            
            # Process each email
            results = []
            for idx, row in emails_df.iterrows():
                result = detector.process_email(row['sender'], row['subject'], row['body'])
                
                # Add results
                emails_df.at[idx, 'is_phishing'] = result['is_phishing']
                emails_df.at[idx, 'phishing_probability'] = result['phishing_probability']
                emails_df.at[idx, 'confidence'] = result['confidence']
                
                results.append({
                    'file_idx': idx,
                    'sender': row['sender'],
                    'subject': row['subject'],
                    'is_phishing': result['is_phishing'],
                    'probability': result['phishing_probability'],
                    'confidence': result['confidence']
                })
            
            # Save detailed results
            output_path = "phishing_analysis_results.csv"
            emails_df.to_csv(output_path, index=False)
            print(f"Analysis complete. Results saved to {output_path}")
            
            # Print summary
            print("\nAnalysis Summary:")
            print(f"Total emails analyzed: {len(emails_df)}")
            phishing_count = emails_df['is_phishing'].sum()
            print(f"Phishing emails detected: {phishing_count} ({phishing_count/len(emails_df)*100:.1f}%)")
            print(f"Legitimate emails: {len(emails_df) - phishing_count} ({(len(emails_df) - phishing_count)/len(emails_df)*100:.1f}%)")
            
            # Print top phishing emails
            print("\nTop 5 Most Likely Phishing Emails:")
            phishing_df = emails_df[emails_df['is_phishing'] == True].sort_values('phishing_probability', ascending=False)
            for idx, row in phishing_df.head(5).iterrows():
                print(f"- From: {row['sender'][:30]}")
                print(f"  Subject: {row['subject'][:50]}")
                print(f"  Probability: {row['phishing_probability']:.2f}, Confidence: {row['confidence']}")
        
        if args.predict:
            if not os.path.exists(model_path):
                print("Model not found. Please train the model first.")
                return
            
            if not args.sender or not args.subject or not args.body:
                print("Please provide sender, subject, and body for prediction.")
                return
            
            detector = PhishingDetector(model_path)
            result = detector.process_email(args.sender, args.subject, args.body)
            
            print("\nPhishing Detection Results:")
            print(f"Verdict: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'}")
            print(f"Confidence: {result['confidence']}")
            print(f"Phishing Probability: {result['phishing_probability']:.2f}")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()