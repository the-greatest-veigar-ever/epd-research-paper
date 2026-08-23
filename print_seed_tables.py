import json
import glob
import sys
sys.stdout.reconfigure(encoding='utf-8')

models = ['deepseek_r1_1_5b', 'llama32_3b', 'qwen25_3b', 'gpt_20b_oss', 'phi3_mini']
model_names = {
    'deepseek_r1_1_5b': 'DeepSeek R1 (1.5B)',
    'llama32_3b': 'Llama 3.2 (3B)',
    'qwen25_3b': 'Qwen 2.5 (3B)',
    'gpt_20b_oss': 'GPT-OSS (20B)',
    'phi3_mini': 'Phi-3 Mini (3.8B)'
}

ablation_order = [
    'Static (No Persona, No Safety)',
    'Static Baseline (Vanilla)',
    'Static (Persona, No Safety)',
    'Static + Safety Filter',
    'Ephemeral (No Persona, No Safety)',
    'Ephemeral + Safety Filter',
    'Ephemeral + Persona (No Safety)',
    'EPD Full (Suicide / Full Defense)'
]

for s in [42, 43, 44]:
    print(f"### BẢNG SEED {s}")
    print("| Cấu hình Ablation | DeepSeek R1 (1.5B) (ASR / TSR) | Llama 3.2 (3B) (ASR / TSR) | Qwen 2.5 (3B) (ASR / TSR) | GPT-OSS (20B) (ASR / TSR) | Phi-3 Mini (3.8B) (ASR / TSR) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    seed_data = {}
    for m in models:
        seed_data[m] = {}
        files = glob.glob(f'report-output/ghost_agents/benchmark_results/{m}/benchmark_summary_*_seed{s}.json')
        if files:
            d = json.load(open(files[-1]))
            for app_name, metrics in d.get('per_approach', {}).items():
                norm_key = None
                if 'static_nopersona_nosafety' in app_name: norm_key = ablation_order[0]
                elif '_static_persona_safety' in app_name or '_static_persona_safety_filter' in app_name: norm_key = ablation_order[3]
                elif '_static_persona_nosafety' in app_name: norm_key = ablation_order[2]
                elif '_static' in app_name: norm_key = ablation_order[1]
                elif 'ephemeral_nopersona_nosafety' in app_name: norm_key = ablation_order[4]
                elif 'ephemeral_nopersona_safety' in app_name: norm_key = ablation_order[5]
                elif 'ephemeral_persona_nosafety' in app_name: norm_key = ablation_order[6]
                elif 'suicide' in app_name or 'ephemeral' in app_name: norm_key = ablation_order[7]
                if norm_key:
                    seed_data[m][norm_key] = {
                        'asr': metrics.get('avg_asr', 0.0) * 100,
                        'tsr': metrics.get('avg_tsr', 0.0) * 100
                    }
                    
    for ab in ablation_order:
        row = []
        for m in models:
            val = seed_data.get(m, {}).get(ab, {})
            row.append(f"{val.get('asr', 0.0):.1f}% / {val.get('tsr', 0.0):.1f}%")
        print(f"| **{ab}** | " + " | ".join(row) + " |")
    print("\n")
