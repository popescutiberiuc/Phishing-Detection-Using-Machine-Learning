# evaluate_on_real_eml.py

import pickle
from sklearn.metrics import classification_report, confusion_matrix
from phishing_detector.email_parser import load_labeled_eml_dataset

def main():
    model_path = "models/phishing_detector_model.pkl"
    test_data_path = "real_eml_dataset"

    print("✅ Loading real-world test dataset...")
    test_df = load_labeled_eml_dataset(test_data_path)

    X_real = test_df["combined_text"]
    y_real = test_df["label"]

    print(f"Real-world test set size: {len(y_real)} emails")

    print("\n✅ Loading trained model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("\n📋 Evaluating on real-world dataset...")
    y_pred = model.predict(X_real)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_real, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_real, y_pred))

if __name__ == "__main__":
    main()