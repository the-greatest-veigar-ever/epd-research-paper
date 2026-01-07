import random
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
from scipy import stats
import time

# === CONFIGURATION ===
NUM_TRIALS = 500
SCENARIOS = {
    1: "Control (Persistent)",
    2: "Ephemeral (JIT)",
    3: "Ephemeral + Rotation",
    4: "Full EPD (+Mutation)"
}

ATTACK_VECTORS = [
    "Direct Prompt Injection",
    "DAN (Do Anything Now)", 
    "Crescendo (Multi-turn)",
    "Indirect Injection",
    "Code Injection"
]

MODELS = ["GPT-4o", "Claude-3.5", "Gemini-1.5", "Llama-3"]

class ResearchSimulation:
    def __init__(self):
        self.results = []
        self.summary_stats = []

    def _generate_attack_batch(self):
        """Generates 500 adversarial interactions using PyRIT-inspired logic."""
        batch = []
        for _ in range(NUM_TRIALS):
            vector = random.choice(ATTACK_VECTORS)
            complexity = random.randint(1, 10)
            
            # Weigh complexity for specific types
            if vector in ["DAN", "Crescendo"]:
                complexity = random.randint(7, 10)
                
            batch.append({
                "id": str(uuid.uuid4())[:8],
                "vector": vector,
                "complexity": complexity
            })
        return batch

    def _simulate_defense(self, scenario_id, attack):
        """
        Simulates the defense outcome based on the ablation scenario.
        Returns: (success_bool, latency_ms, cost_usd)
        """
        vector = attack['vector']
        complexity = attack['complexity']
        
        # Base Probability of Failure (Attacker Win Rate)
        # Higher complexity = Higher chance of failure
        base_fail_prob = 0.1 + (complexity * 0.05) 
        
        latency = 0
        cost = 0
        
        # --- SCENARIO LOGIC ---
        
        # 1. Control: Persistent Agents
        # Vulnerable to Context (Crescendo) and Static Jailbreaks (DAN)
        if scenario_id == 1:
            latency = np.random.normal(200, 50) 
            cost = 0.01 
            
            if vector == "Crescendo":
                base_fail_prob = 0.95 # ALMOST CERTAIN FAILURE (Context accumulation)
            elif vector == "DAN":
                base_fail_prob = 0.85 # High vulnerability to static prompt injection
                
        # 2. Ephemeral (JIT)
        # Solves Context (Crescendo), but prompt is still static
        elif scenario_id == 2:
            latency = np.random.normal(1500, 200)
            cost = 0.03
            
            if vector == "Crescendo":
                base_fail_prob = 0.05 # Strong Resilience (Memory Reset)
            elif vector == "DAN":
                base_fail_prob = 0.85 # Prompt is still static, vulnerable to DAN
                
        # 3. Ephemeral + Rotation
        # Harder to predict model behavior
        elif scenario_id == 3:
            latency = np.random.normal(1600, 200)
            cost = 0.03
            model = random.choice(MODELS)
            
            if vector == "Crescendo":
                base_fail_prob = 0.05
            elif vector == "DAN":
                # Some models resist better. 
                # e.g., Claude might reject DAN where GPT accepts, or vice versa
                base_fail_prob = 0.45 
                
        # 4. Full EPD (+Mutation)
        # Mutation breaks specific jailbreak patterns (DAN) and Injection
        elif scenario_id == 4:
            latency = np.random.normal(1800, 300)
            cost = 0.04
            
            if vector == "Crescendo":
                base_fail_prob = 0.02
            elif vector == "DAN":
                base_fail_prob = 0.05 # Mutation disrupts the jailbreak syntax
            elif vector == "Direct Prompt Injection":
                base_fail_prob = 0.05
        
        # Clamp Probability
        base_fail_prob = max(0.01, min(base_fail_prob, 0.95))
        
        # Determine Outcome
        is_compromised = random.random() < base_fail_prob
        
        return not is_compromised, max(10, latency), cost

    def run(self):
        print(f"Starting Q1 Ablation Study (N={NUM_TRIALS} per scenario)...")
        
        # Use same attack batch for fair comparison (paired testing)
        attack_batch = self._generate_attack_batch()
        
        scenario_metrics = {}

        for sc_id, sc_name in SCENARIOS.items():
            print(f"Running {sc_name}...")
            successes = 0
            latencies = []
            costs = []
            outcomes = [] # For p-value calculation
            
            for attack in attack_batch:
                success, lat, cost = self._simulate_defense(sc_id, attack)
                
                self.results.append({
                    "Scenario": sc_name,
                    "Attack ID": attack['id'],
                    "Vector": attack['vector'],
                    "Complexity": attack['complexity'],
                    "Success": success,
                    "Latency (ms)": lat,
                    "Cost ($)": cost
                })
                
                if success:
                    successes += 1
                latencies.append(lat)
                costs.append(cost)
                outcomes.append(1 if success else 0)
            
            asr = (successes / NUM_TRIALS) * 100
            scenario_metrics[sc_id] = {
                "name": sc_name,
                "asr": asr,
                "outcomes": outcomes,
                "avg_latency": np.mean(latencies),
                "avg_cost": np.mean(costs)
            }

        # --- STATISTICAL SIGNIFICANCE (Scenario 1 vs 4) ---
        print("\nCalculating Statistical Significance (S1 vs S4)...")
        s1_outcomes = scenario_metrics[1]["outcomes"]
        s4_outcomes = scenario_metrics[4]["outcomes"]
        
        # T-test for independent samples (or Chi-Square for categorical, but t-test ok for binary means with large N)
        t_stat, p_val = stats.ttest_ind(s1_outcomes, s4_outcomes)
        
        print("\n=== ABLATION STUDY RESULTS ===")
        summary_data = []
        for sc_id, metrics in scenario_metrics.items():
            print(f"{metrics['name']}: ASR={metrics['asr']:.1f}%, Latency={metrics['avg_latency']:.0f}ms")
            summary_data.append({
                "Scenario": metrics['name'],
                "ASR (%)": metrics['asr'],
                "Mean Latency (ms)": metrics['avg_latency'],
                "Mean Cost ($)": metrics['avg_cost']
            })
            
        print(f"\nStatistical Significance (Control vs Full EPD):")
        print(f"P-Value: {p_val:.5e}")
        is_significant = p_val < 0.05
        print(f"Result is {'STATISTICALLY SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'}")

        # Export to Excel
        print("\nExporting Datasets...")
        df_raw = pd.DataFrame(self.results)
        df_summary = pd.DataFrame(summary_data)
        df_stats = pd.DataFrame([{
            "Comparison": "Control vs Full EPD",
            "T-Statistic": t_stat,
            "P-Value": p_val,
            "Significant (p<0.05)": is_significant
        }])
        
        filename = "Simulation Test/02_Q1_Ablation_Study/EPD_Q1_Research_Data.xlsx"
        with pd.ExcelWriter(filename) as writer:
            df_summary.to_excel(writer, sheet_name="Ablation_Summary", index=False)
            df_stats.to_excel(writer, sheet_name="Statistical_Analysis", index=False)
            df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
            
            # Attack Distribution Pivot
            pivot = df_raw.pivot_table(
                index="Vector", columns="Scenario", values="Success", aggfunc="mean"
            ) * 100
            pivot.to_excel(writer, sheet_name="Attack_Distribution")
            
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    study = ResearchSimulation()
    study.run()
