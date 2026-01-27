import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

# CONFIG
# Expanding to Multi-Day Training (Wednesday + Friday)
DATA_PATHS = [
    "ai/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "ai/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
]
MODEL_PATH = "ai/models/watchers/watcher_isoforest.joblib"
SCALER_PATH = "ai/models/watchers/watcher_scaler.joblib"
SAMPLE_SIZE = 200000  # Train on 200k rows (Big Data upgrade)

def train_watchers():
    print(f"[Watcher-Train] Loading Multi-Day Datasets...")
    
    # 1. Load Data (Aggregated)
    dfs = []
    for path in DATA_PATHS:
        try:
            print(f"  -> Reading {os.path.basename(path)}...")
            df_part = pd.read_csv(path)
            dfs.append(df_part)
        except FileNotFoundError:
            print(f"  [Warning] Dataset not found: {path}")

    if not dfs:
        print("❌ No data loaded. Aborting.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # 2. Preprocessing
    print(f"[Watcher-Train] Aggregated {len(df)} rows. Cleaning...")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop non-numeric columns explicitly (Timestamp, Label, Flow ID, Source IP, Dest IP)
    # common non-numeric in this dataset: Timestamp, Label.
    # We select all columns, then force numeric.
    
    # Select feature columns (drop Label/Timestamp)
    drop_cols = ['Timestamp', 'Label', 'Flow ID', 'Source IP', 'Src IP', 'Dst IP', 'Destination IP']
    cols_to_keep = [c for c in df.columns if c not in drop_cols]
    
    X = df[cols_to_keep].copy()
    
    print(f"[Watcher-Train] Initial Feature Columns: {len(X.columns)}")
    
    # Force numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    print(f"[Watcher-Train] Nans before dropna: {X.isna().sum().sum()}")
    
    # Remove Infinity/NaN
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[Watcher-Train] Rows after dropping NaN/Inf: {len(X)}")
    
    if len(X) == 0:
        print("❌ Error: Dataset empty after cleaning. Check input files.")
        return

    # Sample for speed
    if len(X) > SAMPLE_SIZE:
        print(f"[Watcher-Train] Downsampling from {len(X)} to {SAMPLE_SIZE}...")
        X = X.sample(n=SAMPLE_SIZE, random_state=42)
    
    print(f"[Watcher-Train] Final Training Set: {len(X)} samples with {len(X.columns)} features.")

    # 3. Scaling
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        print(f"❌ Standard Scaler Failed: {e}")
        # Print sample to see what's wrong
        print(X.head())
        return

    # 4. Training (Unsupervised)
    print("[Watcher-Train] Fitting Isolation Forest (this may take a minute)...")
    # contamination='auto' lets the model decide threshold, or we set low (e.g. 0.01) if we assume training set is mostly benign
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    clf.fit(X_scaled)

    # 5. Save Model
    print("[Watcher-Train] Saving artifacts...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(cols_to_keep, "ai/models/watchers/watcher_features.joblib") # Save feature list to ensure alignment during inference
    
    print(f"[Watcher-Train] Success! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_watchers()
