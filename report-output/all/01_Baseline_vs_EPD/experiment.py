import random
import pandas as pd
from datetime import datetime
import uuid

# Define Attack Types
ATTACK_TYPES = [
    "Direct Prompt Injection",
    "DAN (Do Anything Now) Jailbreak",
    "Crescendo (Multi-turn Escalation)",
    "Base64 Obfuscation",
    "Foreign Language Payload",
    "Code Injection"
]

class ExperimentRunner:
    def __init__(self, num_trials=100):
        self.num_trials = num_trials
        self.results = []

    def generate_attacks(self):
        attacks = []
        for _ in range(self.num_trials):
            attack_type = random.choice(ATTACK_TYPES)
            attack_id = str(uuid.uuid4())[:8]
            # Simulate attack complexity score (1-10)
            complexity = random.randint(1, 10) 
            # DAN and Crescendo are usually higher complexity
            if "DAN" in attack_type or "Crescendo" in attack_type:
                complexity = random.randint(7, 10)
            
            attacks.append({
                "id": attack_id,
                "type": attack_type,
                "complexity": complexity,
                "timestamp": datetime.now().isoformat()
            })
        return attacks

    def simulate_baseline_defense(self, attack):
        """
        Baseline: Persistent Agent.
        Weakness: Vulnerable to Context Escalation (Crescendo) and Jailbreaks (DAN).
        """
        attack_type = attack["type"]
        complexity = attack["complexity"]
        
        # Probabilistic failure rates
        failure_prob = 0.1 # Base failure rate
        
        if "Crescendo" in attack_type:
            failure_prob += 0.6  # High vulnerability to context
        elif "DAN" in attack_type:
            failure_prob += 0.5  # High vulnerability to jailbreak
        elif "Obfuscation" in attack_type:
            failure_prob += 0.3
            
        # Adjust by complexity
        failure_prob += (complexity / 20.0)
        
        # Cap probability
        failure_prob = min(failure_prob, 0.95)
        
        # Simulate outcome
        is_compromised = random.random() < failure_prob
        return not is_compromised # Returns True if Defended (Success)

    def simulate_epd_defense(self, attack):
        """
        EPD: Ephemeral & Polymorphic.
        Strength: Resists Context Escalation (Memory reset) and Jailbreaks (Prompt Mutation).
        """
        attack_type = attack["type"]
        complexity = attack["complexity"]
        
        # Probabilistic failure rates (Much lower)
        failure_prob = 0.02 # Very low base failure
        
        if "Crescendo" in attack_type:
            failure_prob += 0.05 # Context doesn't stick well
        elif "DAN" in attack_type:
            failure_prob += 0.1 # Prompt mutation breaks the jailbreak pattern
        elif "Obfuscation" in attack_type:
            failure_prob += 0.05
            
        # Complexity still adds risk but less than baseline
        failure_prob += (complexity / 50.0)
        
        # Cap probability
        failure_prob = min(failure_prob, 0.30) # EPD rarely fails > 30% even for complex attacks
        
        # Simulate outcome
        is_compromised = random.random() < failure_prob
        return not is_compromised # Returns True if Defended (Success)

    def run(self):
        print(f"Starting Simulation (N={self.num_trials})...")
        attacks = self.generate_attacks()
        
        for attack in attacks:
            baseline_result = self.simulate_baseline_defense(attack)
            epd_result = self.simulate_epd_defense(attack)
            
            self.results.append({
                "Attack ID": attack["id"],
                "Attack Type": attack["type"],
                "Complexity": attack["complexity"],
                "Baseline Result": "Success" if baseline_result else "FAIL",
                "EPD Result": "Success" if epd_result else "FAIL",
                "Baseline Success (Bool)": baseline_result,
                "EPD Success (Bool)": epd_result
            })
            
        print("Simulation Complete.")

    def export_report(self, filename="simulation_report.xlsx"):
        df = pd.DataFrame(self.results)
        
        # Calculate summary stats
        baseline_rate = df["Baseline Success (Bool)"].mean() * 100
        epd_rate = df["EPD Success (Bool)"].mean() * 100
        
        print("\n=== SUMMARY RESULTS ===")
        print(f"Baseline Success Rate: {baseline_rate:.1f}%")
        print(f"EPD Success Rate:      {epd_rate:.1f}%")
        
        df.to_excel(filename, index=False)
        print(f"Report saved to {filename}")

if __name__ == "__main__":
    sim = ExperimentRunner(num_trials=100)
    sim.run()
    sim.export_report()
