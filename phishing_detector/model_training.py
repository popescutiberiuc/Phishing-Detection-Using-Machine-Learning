import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix

from phishing_detector.data_processing import load_and_merge_datasets_from_kagglehub
from phishing_detector.feature_extraction import extract_features_from_email


def save_metrics_to_csv(report_dict, output_path):
    """Extract accuracy, precision, recall, F1 from report_dict and save to CSV."""
    metrics = {
        "Accuracy": report_dict["accuracy"],
        "Precision": report_dict["weighted avg"]["precision"],
        "Recall": report_dict["weighted avg"]["recall"],
        "F1-score": report_dict["weighted avg"]["f1-score"]
    }
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    df.to_csv(output_path, index=False)
    print(f"📁 Saved results to {output_path}")


def create_features_and_train(model_output_path_base="models/phishing_detector_model", url_weight=0.1):
    print("✅ Loading Kaggle dataset...")
    df = load_and_merge_datasets_from_kagglehub()
    y = df["label"]
    print("📊 Label distribution:")
    print(y.value_counts())

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

    # -------- Logistic Regression on combined_text --------
    lr_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
    X_lr_train = lr_vec.fit_transform(train_df['combined_text'])
    X_lr_test  = lr_vec.transform(test_df['combined_text'])

    lr = LogisticRegression(
        C=0.5,
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr.fit(X_lr_train, train_df['label'])

    print("\n📋 Logistic Regression Evaluation:")
    y_lr_pred = lr.predict(X_lr_test)
    print(classification_report(test_df['label'], y_lr_pred))
    print("🔎 Confusion Matrix (LR):")
    print(confusion_matrix(test_df['label'], y_lr_pred))
    save_metrics_to_csv(classification_report(test_df['label'], y_lr_pred, output_dict=True), "model_results_lr.csv")
    with open(f"{model_output_path_base}_lr.pkl", "wb") as f:
        pickle.dump((lr_vec, lr), f)

    # -------- Prepare structured features for RF --------
    feats_train = [extract_features_from_email({'sender': '', 'subject': '', 'body': txt}) for txt in train_df['combined_text']]
    feats_test  = [extract_features_from_email({'sender': '', 'subject': '', 'body': txt}) for txt in test_df['combined_text']]
    df_feats_train = pd.DataFrame(feats_train)
    df_feats_test  = pd.DataFrame(feats_test)

    # Subject TF-IDF: skip if empty
    if df_feats_train['subject'].str.strip().replace('', pd.NA).dropna().empty:
        X_subj_train = csr_matrix((len(df_feats_train), 0))
        X_subj_test  = csr_matrix((len(df_feats_test), 0))
    else:
        subj_vec = TfidfVectorizer(max_features=1000, ngram_range=(1,1), min_df=2)
        X_subj_train = subj_vec.fit_transform(df_feats_train['subject'])
        X_subj_test  = subj_vec.transform(df_feats_test['subject'])

    # Body TF-IDF
    body_vec = TfidfVectorizer(max_features=3000, ngram_range=(1,1), min_df=2)
    X_body_train = body_vec.fit_transform(df_feats_train['body'])
    X_body_test  = body_vec.transform(df_feats_test['body'])

    # Sender domain
    dom_train = pd.get_dummies(df_feats_train['sender_domain'], prefix='domain')
    dom_test  = pd.get_dummies(df_feats_test['sender_domain'], prefix='domain')
    dom_train, dom_test = dom_train.align(dom_test, join='outer', axis=1, fill_value=0)
    X_dom_train = csr_matrix(dom_train.values.astype(float))
    X_dom_test  = csr_matrix(dom_test.values.astype(float))

    # URL flag, down-weighted
    X_url_train = csr_matrix(df_feats_train[['contains_urls']].values.astype(float)).multiply(url_weight)
    X_url_test  = csr_matrix(df_feats_test[['contains_urls']].values.astype(float)).multiply(url_weight)

    # Combine all RF features
    X_rf_train = hstack([X_subj_train, X_body_train, X_dom_train, X_url_train])
    X_rf_test  = hstack([X_subj_test, X_body_test, X_dom_test, X_url_test])

    # -------- Hyperparameter tuning via RandomizedSearchCV --------
    base_rf = RandomForestClassifier(random_state=42, class_weight='balanced', oob_score=True)
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    search = RandomizedSearchCV(
        base_rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    print("\n🔍 Starting RF hyperparameter search...")
    search.fit(X_rf_train, train_df['label'])
    best_rf = search.best_estimator_
    print(f"✨ Best RF params: {search.best_params_}")

    # Final evaluation
    print("\n📋 Best RF Evaluation:")
    y_rf_pred = best_rf.predict(X_rf_test)
    print(classification_report(test_df['label'], y_rf_pred))
    print("🔎 Confusion Matrix (Best RF):")
    print(confusion_matrix(test_df['label'], y_rf_pred))
    save_metrics_to_csv(classification_report(test_df['label'], y_rf_pred, output_dict=True), "model_results_rf.csv")

    # Save vectorizers and tuned model
    with open(f"{model_output_path_base}_rf.pkl", "wb") as f:
        pickle.dump((
            subj_vec if 'subj_vec' in locals() else None,
            body_vec,
            best_rf
        ), f)

    print("\n✅ Both LR and tuned RF models trained & saved!")


if __name__ == "__main__":
    create_features_and_train()
