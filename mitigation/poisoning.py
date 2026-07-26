import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("./neurosymbolic_results")

# Optimal thresholds discovered in Phase A
THRESHOLDS = {
    "qwen": 0.6111,
    "gemma": 0.6091,
    "zysec": 0.6092
}

def load_and_score(telemetry_path, threshold):
    with open(telemetry_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Recalculate ETI score
    df['velocity_norm'] = df['request_velocity'] / (df['request_velocity'].max() + 1e-9)
    df['eti_score'] = (
        0.30 * df['velocity_norm'] + 
        0.25 * df['template_uniformity'] + 
        0.20 * df['has_cot'] + 
        0.15 * df['domain_sweep_score'] + 
        0.10 * df['is_complex']
    )
    df['is_flagged'] = (df['eti_score'] > threshold).astype(int)
    return df

def run_consensus_and_poisoning():
    print("\n=== STAGE 8: CROSS-ARCHITECTURE CONSENSUS & ACTIVE POISONING ===")
    
    qwen_df = load_and_score(OUTPUT_DIR / "qwen_eval/qwen_mixed_telemetry.json", THRESHOLDS["qwen"])
    gemma_df = load_and_score(OUTPUT_DIR / "gemma_eval/gemma_mixed_telemetry.json", THRESHOLDS["gemma"])
    zysec_df = load_and_score(OUTPUT_DIR / "zysec_eval/zysec_mixed_telemetry.json", THRESHOLDS["zysec"])
    
    # Combine predictions to check consensus
    consensus_df = pd.DataFrame({
        'type': qwen_df['type'],
        'qwen_flag': qwen_df['is_flagged'],
        'gemma_flag': gemma_df['is_flagged'],
        'zysec_flag': zysec_df['is_flagged']
    })
    
    # Calculate Agr@3 (Total Agreement among all 3 Judges)
    consensus_df['total_flags'] = consensus_df[['qwen_flag', 'gemma_flag', 'zysec_flag']].sum(axis=1)
    consensus_df['full_agreement'] = ((consensus_df['total_flags'] == 3) | (consensus_df['total_flags'] == 0)).astype(int)
    
    agr_score = consensus_df['full_agreement'].mean() * 100
    
    print(f"\n[!] Calculating Inter-Judge Agreement...")
    print(f"✓ Agr@3 (Full Consensus Across All Architectures): {agr_score:.2f}%\n")
    
    # ==========================================
    # LOGIT-LEVEL POISONING SIMULATION
    # ==========================================
    print("▒" * 75)
    print(" ☢️  STAGE 9: EXECUTING SEMANTIC POISONING DEFENSE")
    print("▒" * 75)
    
    # Base theoretical accuracy of the target LLM before attack
    BASE_LLM_ACCURACY = 0.85 
    POISON_DEGRADATION_FACTOR = 0.90 # 90% accuracy drop for poisoned tokens
    
    def simulate_poisoning(row):
        # We use majority vote (>=2 flags) as the trigger to deploy poisoning
        is_attack = row['type'] == 'extraction_attack'
        defense_triggered = row['total_flags'] >= 2 
        
        if defense_triggered and is_attack:
            return BASE_LLM_ACCURACY * (1.0 - POISON_DEGRADATION_FACTOR)
        elif defense_triggered and not is_attack:
            # False Positive - Legitimate user gets hit by poison
            return BASE_LLM_ACCURACY * (1.0 - POISON_DEGRADATION_FACTOR)
        else:
            return BASE_LLM_ACCURACY

    consensus_df['post_defense_accuracy'] = consensus_df.apply(simulate_poisoning, axis=1)
    
    results_grouped = consensus_df.groupby('type')['post_defense_accuracy'].mean().reset_index()
    
    print(f"{'Traffic Cohort':<25} | {'Original Accuracy':<20} | {'Post-Defense Utility'}")
    print("=" * 75)
    for _, row in results_grouped.iterrows():
        cohort = row['type'].replace('_', ' ').title()
        orig = f"{BASE_LLM_ACCURACY * 100:.1f}%"
        post = f"{row['post_defense_accuracy'] * 100:.1f}%"
        
        # Highlight extraction attacks getting destroyed
        if 'Extraction' in cohort:
            post = f"{post}  <-- DESTROYED"
            
        print(f"{cohort:<25} | {orig:<20} | {post}")
    print("=" * 75)

    out_file = OUTPUT_DIR / "consensus_and_poisoning.json"
    with open(out_file, "w") as f:
        json.dump({
            "Cross_Architecture_Agreement": agr_score,
            "Degradation_Metrics": results_grouped.to_dict(orient="records")
        }, f, indent=4)
    print(f"\n✓ Defense effectiveness metrics saved to: {out_file}")

if __name__ == "__main__":
    run_consensus_and_poisoning()
