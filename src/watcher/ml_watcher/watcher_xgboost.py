import pandas as pd
import joblib
import os
import time
import numpy as np
from datetime import datetime

class XGBoostWatcher:
    def __init__(self, model_path="models/xgboost_watcher.joblib"):
        self.model = joblib.load(model_path)
        print(f"[XGBoostWatcher] Loaded model from {model_path}")
        
        # Define the enriched feature set (21 features)
        self.features_list = [
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

    def predict(self, log_entry):
        """
        Expects a log entry dictionary with enriched behavioral metadata.
        """
        start_time = time.perf_counter()
        
        # 1. Feature Extraction / Mapping
        data = {
            "duration_minutes": log_entry.get("duration_minutes", 0),
            "Dst Port": log_entry.get("dst_port", 0),
            "Protocol": log_entry.get("protocol", 6),
            "Tot Fwd Pkts": log_entry.get("tot_fwd_pkts", 0),
            "Tot Bwd Pkts": log_entry.get("tot_bwd_pkts", 0),
            "TotLen Fwd Pkts": log_entry.get("tot_len_fwd_pkts", 0),
            "TotLen Bwd Pkts": log_entry.get("tot_len_bwd_pkts", 0),
            "Flow Byts/s": log_entry.get("flow_byts_s", 0),
            "Flow Pkts/s": log_entry.get("flow_pkts_s", 0),
            "Flow IAT Mean": log_entry.get("flow_iat_mean", 0),
            "Flow IAT Max": log_entry.get("flow_iat_max", 0),
            "Flow IAT Min": log_entry.get("flow_iat_min", 0),
            "FIN Flag Cnt": log_entry.get("fin_flag_cnt", 0),
            "SYN Flag Cnt": log_entry.get("syn_flag_cnt", 0),
            "RST Flag Cnt": log_entry.get("rst_flag_cnt", 0),
            "PSH Flag Cnt": log_entry.get("psh_flag_cnt", 0),
            "ACK Flag Cnt": log_entry.get("ack_flag_cnt", 0),
            "Init Fwd Win Byts": log_entry.get("init_fwd_win_byts", 0),
            "Init Bwd Win Byts": log_entry.get("init_bwd_win_byts", 0),
        }
        
        # Handle Time
        if "date" in log_entry and "start_time" in log_entry:
            try:
                date_str = "-".join(log_entry["date"].split("-")[1:])
                start = datetime.strptime(f"{date_str} {log_entry['start_time']}", "%d-%m-%Y %H:%M")
                data["hour_of_day"] = start.hour
                data["day_of_week"] = start.weekday()
            except:
                data["hour_of_day"] = 0
                data["day_of_week"] = 0
        else:
            data["hour_of_day"] = log_entry.get("hour_of_day", 0)
            data["day_of_week"] = log_entry.get("day_of_week", 0)

        # Create DataFrame matching training columns
        features_df = pd.DataFrame([data])[self.features_list]
        
        # 2. Inference
        prediction = self.model.predict(features_df)[0]
        probability = self.model.predict_proba(features_df)[0][1]
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        return prediction, probability, inference_time_ms

    def predict_batch(self, df):
        """
        Batch inference for high-speed traffic scanning.
        Expects a DataFrame that already contains the necessary columns (mostly).
        Calculates missing columns if needed.
        """
        # Ensure we have the right features
        # 1. Map columns if they don't match perfectly (CICFlowMeter has different casing sometimes)
        # For simplicity in this demo, we assume the CSV cleaning in runner handles most, 
        # but we ensure existence.
        
        # We need to ensure the DF has the columns self.features_list expects.
        # Common mapping from CICFlowMeter CSV headers to our training headers:
        column_map = {
             "Dst Port": "Dst Port",
             "Protocol": "Protocol",
             "Tot Fwd Pkts": "Tot Fwd Pkts",
             "Tot Bwd Pkts": "Tot Bwd Pkts",
             "TotLen Fwd Pkts": "TotLen Fwd Pkts",
             "TotLen Bwd Pkts": "TotLen Bwd Pkts",
             "Flow Byts/s": "Flow Byts/s",
             "Flow Pkts/s": "Flow Pkts/s",
             "Flow IAT Mean": "Flow IAT Mean",
             "Flow IAT Max": "Flow IAT Max",
             "Flow IAT Min": "Flow IAT Min",
             "Fwd IAT Tot": "duration_minutes", # Approx mapping for duration
             "FIN Flag Cnt": "FIN Flag Cnt",
             "SYN Flag Cnt": "SYN Flag Cnt",
             "RST Flag Cnt": "RST Flag Cnt",
             "PSH Flag Cnt": "PSH Flag Cnt",
             "ACK Flag Cnt": "ACK Flag Cnt",
             "Init Fwd Win Byts": "Init Fwd Win Byts",
             "Init Bwd Win Byts": "Init Bwd Win Byts"
        }
        
        X = pd.DataFrame()
        for feature in self.features_list:
            if feature in df.columns:
                X[feature] = df[feature]
            elif feature == "duration_minutes" and "Flow Duration" in df.columns:
                 # Flow Duration is usually in Microseconds in CIC datasets
                 X[feature] = df["Flow Duration"] / 60000000.0
            elif feature == "hour_of_day":
                # Timestamp parsing is slow, fill with 0 for batch speed or parse if 'Timestamp' exists
                X[feature] = 12 # Default
            elif feature == "day_of_week":
                X[feature] = 2 # Default
            else:
                X[feature] = 0.0 # Fill missing features
        
        # Fill NaNs
        X = X.fillna(0)
        
        # Predict
        # model.predict returns 0 (benign) / 1 (malicious) usually for XGBoost classifiers
        # But IsolationForest was -1/1. We need to check the model type.
        # Assuming XGBoost trained as binary classifier 0/1.
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        
        return preds, probs
