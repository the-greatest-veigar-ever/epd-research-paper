import time
from src.monitor import DetectionAgent

def test_watcher():
    print("--- Starting Watcher Verification (Optimized Recall) ---")
    watcher = DetectionAgent("watcher-test")
    
    # Test case 1: Verified HOIC Attack with full metadata
    test_logs = [
        {
            "src_ip": "18.218.115.60", 
            "dst_ip": "172.31.69.25", 
            "attack_name": "DDOS attack-HOIC", 
            "date": "Wed-21-02-2018", 
            "start_time": "02:11", 
            "end_time": "02:12",
            "dst_port": 80,
            "protocol": 6,
            "tot_fwd_pkts": 3,
            "tot_bwd_pkts": 4,
            "tot_len_fwd_pkts": 248,
            "tot_len_bwd_pkts": 935,
            "flow_byts_s": 69625,
            "flow_pkts_s": 411,
            "syn_flag_cnt": 1,
            "push_flag_cnt": 1,
            "init_fwd_win_byts": 26883,
            "event_name": "Verified_HOIC_Attack"
        },
        {
            "src_ip": "172.31.70.4", 
            "dst_ip": "172.31.69.25", 
            "attack_name": "Benign", 
            "date": "Wed-14-02-2018", 
            "start_time": "10:32", 
            "dst_port": 443,
            "protocol": 6,
            "flow_byts_s": 100,
            "tot_fwd_pkts": 2,
            "event_name": "Regular_Traffic"
        }
    ]
    
    print("\nScanning enriched test logs...")
    alerts = watcher.monitor_logs(test_logs)
    print(f"Alerts generated: {len(alerts)}")
    for alert in alerts:
        print(f"Alert: {alert['details'].get('attack_name')} with Risk: {alert.get('risk_score')}")

if __name__ == "__main__":
    test_watcher()
