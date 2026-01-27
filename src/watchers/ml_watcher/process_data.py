import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from sklearn.preprocessing import LabelEncoder
import joblib

# CONFIG Matches the remote branch method
INPUT_DIR = "ai/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms"
OUTPUT_FILE = "ai/data/processed_watcher_data.csv"
SAMPLE_SIZE = 10000000 # Increased to 10 Million (Extreme robustness)

def process_logs(input_dir, output_file, sample_size=SAMPLE_SIZE):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    all_data = []

    print(f"Found {len(csv_files)} CSV files. Processing...")

    for file in csv_files:
        print(f"Reading {os.path.basename(file)}...")
        # Read a smaller sample from each file to avoid memory explosion during concat
        # or read full if memory allows. 
        # Strategy: Read relevant columns only.
        
        # Cols from remote branch
        cols_to_keep = [
            'Timestamp', 'Flow Duration', 'Dst Port', 'Protocol', 
            'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
            'Flow Byts/s', 'Flow Pkts/s', 
            'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
            'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
            'Init Fwd Win Byts', 'Init Bwd Win Byts',
            'Label'
        ]
        
        try:
            # Read header to check cols
            header = pd.read_csv(file, nrows=0)
            header.columns = header.columns.str.strip()
            existing_cols = [c for c in cols_to_keep if c in header.columns]
            
            # Read file with specific columns
            df_chunk = pd.read_csv(file, usecols=existing_cols)
            df_chunk.columns = df_chunk.columns.str.strip() # Re-strip after read
            
            # Subsample immediately to save memory (take 10% of each file or fixed amount)
            # We will take 1,200,000 from each (10 files * 1.2M = 12M -> sample down to 10M later)
            if len(df_chunk) > 1200000:
                 df_chunk = df_chunk.sample(n=1200000, random_state=42)
                 
            all_data.append(df_chunk)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    print("Concatenating...")
    df = pd.concat(all_data, ignore_index=True)
    
    # Numeric Conversion
    numeric_cols = [
        'Flow Duration', 'Dst Port', 'Protocol', 'Tot Fwd Pkts', 
        'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Flow Byts/s', 'Flow Pkts/s',
        'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
        'Init Fwd Win Byts', 'Init Bwd Win Byts'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle Timestamp
    print("Parsing Timestamps...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
    
    # Drop rows with critical missing values
    df = df.dropna(subset=['Timestamp', 'Flow Duration', 'Label'])
    
    # Feature Extraction
    print("Feature Engineering...")
    if 'Flow Duration' in df.columns:
        df['duration_minutes'] = df['Flow Duration'] / (1000000 * 60)
    else:
        df['duration_minutes'] = 0
        
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.weekday
    
    # multi_source_attack
    df['multi_source_attack'] = df['Label'].str.contains("DDoS|Bot", case=False).astype(int)
    
    # Mocks
    df['src_is_internal'] = 1 
    df['dst_is_internal'] = 1
    
    # Label Encoding
    # Ensure directory exists for joblib
    os.makedirs("ai/models/watchers", exist_ok=True)
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Label'])
    df['is_malicious'] = (df['Label'] != 'Benign').astype(int)
    
    joblib.dump(le, 'ai/models/watchers/watcher_label_encoder.joblib')
    
    # Downsample Final
    if len(df) > sample_size:
        print(f"Downsampling from {len(df)} to {sample_size}...")
        df = df.sample(n=sample_size, random_state=42)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # Print Stats
    print("\nDataset Stats:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    process_logs(INPUT_DIR, OUTPUT_FILE)
