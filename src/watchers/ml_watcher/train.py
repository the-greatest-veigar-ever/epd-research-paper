import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, recall_score
import joblib
import os
import time

def train_optimized_model(data_path, model_output_path):
    print(f"Loading enriched data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Preprocessing (Data is already processed by process_data.py)
    # Just ensure types and basic checks
    if "is_malicious" not in df.columns:
        if "Label" in df.columns:
             df['is_malicious'] = (df['Label'] != 'Benign').astype(int)
        else:
             print("Warning: No Label column found. Assuming all benign (0) for test.")
             df['is_malicious'] = 0
         
    # Ensure features exist (fill 0 if missing from processed)
    for feat in ["duration_minutes", "hour_of_day", "day_of_week"]:
        if feat not in df.columns:
            df[feat] = 0
            
    # Deep behavioral features
    features = [
        "duration_minutes",
        "hour_of_day",
        "day_of_week",
        "Dst Port",
        "Protocol",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
        "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
        "Init Fwd Win Byts", "Init Bwd Win Byts"
    ]
    
    # Preprocessing
    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    y = df["is_malicious"]
    
    # Calculate scale_pos_weight for imbalance
    num_benign = (y == 0).sum()
    num_malicious = (y == 1).sum()
    if num_malicious > 0:
        scale_weight = num_benign / num_malicious
    else:
        scale_weight = 1.0
        
    print(f"Addressing class imbalance with scale_pos_weight: {scale_weight:.2f}")
    
    # Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print("\n--- Phase 1: Hyperparameter Tuning (Optimizing for Recall) ---")
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_weight, # Crucial for Recall
        random_state=42,
        tree_method="hist"
    )
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_grid, n_iter=10, 
        scoring='recall', n_jobs=-1, cv=skf, verbose=1, random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    print(f"\nBest Parameters: {best_params}")
    
    print("\n--- Phase 2: Training with Early Stopping ---")
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=15
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_pred = final_model.predict(X_test)
    print("\nFinal Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall (Malicious): {recall_score(y_test, y_pred):.4f}")
    
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save Report to File
    report_file = "report-output/watchers/training_results.txt"
    with open(report_file, "w") as f:
        f.write("=== XGBoost Watcher Training Results ===\n\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Recall (Malicious): {recall_score(y_test, y_pred):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nModel saved to: {model_output_path}\n")
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_model, model_output_path)
    print(f"\nOptimized model saved to {model_output_path}")

if __name__ == "__main__":
    data_path = "ai/data/processed_watcher_data.csv"
    model_output_path = "ai/models/watchers/xgboost_watcher.joblib"
    
    # Check if processed data exists, if not, warn or run process
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run process_data.py first.")
    else:
        train_optimized_model(data_path, model_output_path)
