import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from sklearn.preprocessing import LabelEncoder
import joblib

def process_logs(input_dir, output_file, sample_size=100000):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    all_data = []

    for file in csv_files:
        print(f"Processing {file}...")
        # Read a sample to manage memory
        df_chunk = pd.read_csv(file)
        
        # Clean column names (strip whitespace)
        df_chunk.columns = df_chunk.columns.str.strip()
        
        # We need: Timestamp, Flow Duration, Label
        # The user's features: src_is_internal, dst_is_internal, duration_minutes, hour_of_day, day_of_week, multi_source_attack, attack_type_encoded
        
        # Since processed CSVs might not have IP, we skip src_is_internal/dst_is_internal or mock them
        # For this research, we'll focus on the available features in the CSV
        
        # Filter for relevant behavioral columns (avoiding Label or anything derived from it)
        cols_to_keep = [
            'Timestamp', 'Flow Duration', 'Dst Port', 'Protocol', 
            'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
            'Flow Byts/s', 'Flow Pkts/s', 
            'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
            'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
            'Init Fwd Win Byts', 'Init Bwd Win Byts',
            'Label'
        ]
        
        # Ensure columns exist before filtering
        df_chunk = df_chunk[[c for c in cols_to_keep if c in df_chunk.columns]]
        
        all_data.append(df_chunk)

    df = pd.concat(all_data, ignore_index=True)
    
    # Convert behavioral columns to numeric, handle errors
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
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
    
    # Drop rows with critical missing values
    df = df.dropna(subset=['Timestamp', 'Flow Duration', 'Label'])
    
    # Feature Extraction
    df['duration_minutes'] = df['Flow Duration'] / (1000000 * 60) # Flow Duration is in microseconds
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.weekday
    
    # multi_source_attack: 1 if Label contains "DDoS" or "Bot"
    df['multi_source_attack'] = df['Label'].str.contains("DDoS|Bot", case=False).astype(int)
    
    # Mocking internal IPs as they aren't in these specific CSVs but are in the user's manual logs
    # We'll set them to a default for consistent feature vector size
    df['src_is_internal'] = 1 
    df['dst_is_internal'] = 1
    
    # Label Encoding
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['Label'])
    # Binary label: 0 if Benign, 1 otherwise
    df['is_malicious'] = (df['Label'] != 'Benign').astype(int)
    
    # Save LabelEncoder for later
    joblib.dump(le, 'models/label_encoder.joblib')
    
    # Downsample if needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    input_base_dir = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/watchers/cse-cic-ids2018/Processed Traffic Data for ML Algorithms"
    output_path = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/processed_watcher_data.csv"
    
    # Ensure models directory exists
    os.makedirs("/Users/thachngo/Documents/EDP Research/epd-research-paper/models", exist_ok=True)
    
    process_logs(input_base_dir, output_path)
