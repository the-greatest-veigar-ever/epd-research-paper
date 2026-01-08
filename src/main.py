import time
from monitor import DetectionAgent, generate_attack_logs
from intelligence import IntelligenceAgent
from epd_core import GhostAgentFactory

def run_simulation():
    print("=== INITIALIZING HYBRID MULTI-AGENT DEFENSE ARCHITECTURE (EPD) ===")
    print("Transitioning from 'Persistent Robustness' to 'Ephemeral Polymorphic Defense'...\n")

    # 1. Initialize Persistent Agents
    watcher = DetectionAgent(agent_id="Watcher-Alpha")
    brain = IntelligenceAgent(agent_id="Brain-Core")

    while True:
        print("\n=== HACKER CONSOLE (Red Team Mode) ===")
        print("Select an attack vector to simulate:")
        print("1. IAM Privilege Escalation (Attacker creates admin keys)")
        print("2. S3 Data Exfiltration (Attacker makes bucket public)")
        print("3. EC2 Crypto Mining (Attacker spawns mining rigs)")
        print("4. DDoS Attack (Traffic Spike)")
        print("5. SQL Injection (WAF Evasion)")
        print("6. Ransomware Precursor (Mass Encryption)")
        print("7. Lateral Movement (RDP Brute Force)")
        print("8. EXIT SIMULATION")
        
        choice = input("Enter Command [1-8]: ").strip()
        
        attack_type = ""
        if choice == '1':
            attack_type = "IAM_PRIVILEGE_ESCALATION"
        elif choice == '2':
            attack_type = "S3_DATA_EXFILTRATION"
        elif choice == '3':
            attack_type = "EC2_CRYPTO_MINING"
        elif choice == '4':
            attack_type = "DDoS_ATTACK"
        elif choice == '5':
            attack_type = "SQL_INJECTION"
        elif choice == '6':
            attack_type = "RANSOMWARE_PRECURSOR"
        elif choice == '7':
            attack_type = "LATERAL_MOVEMENT"
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid command.")
            continue

        print(f"\n[+] SIMULATING ATTACK: {attack_type}...")
        time.sleep(1)

        # 2. Simulate Attack Log Stream
        logs = generate_attack_logs(attack_type)
        
        # 3. Detection Phase (Group 1)
        print("\n--- PHASE 1: DETECTION (Persistent) ---")
        alerts = watcher.monitor_logs(logs)
        
        if not alerts:
            print("No threats detected.")
            continue

        for alert in alerts:
            # 4. Intelligence Phase (Group 2)
            print("\n--- PHASE 2: INTELLIGENCE & CONSENSUS (Hybrid) ---")
            remediation_plan = brain.analyze_alert(alert)
            
            if remediation_plan:
                # 5. Remediation Phase (Group 3 - EPD)
                print("\n--- PHASE 3: REMEDIATION (Ephemeral & Polymorphic) ---")
                print("Triggering Just-in-Time Ghost Agent instantiation...")
                
                # Create Ghost Agent (Just-in-Time)
                ghost = GhostAgentFactory.create_agent(
                    base_instructions=f"Perform {remediation_plan['action']}"
                )
                
                # Execute and Die
                ghost.execute_remediation(remediation_plan)
                
                # Verify Self-Destruct
                if not ghost.is_alive:
                    print(f"Verification: Ghost Agent successfully terminated.")
                else:
                    print("WARNING: Ghost Agent failed to terminate!")
        
        print("\n[+] Threat Mitigated. System returning to baseline monitoring.")
    
    print("\n=== SIMULATION COMPLETE ===")

if __name__ == "__main__":
    run_simulation()
