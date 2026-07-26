import os
import gc
import json
import time
import torch
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import logging
from transformers import logging as transformers_logging

try:
    from llama_cpp import Llama, LogitsProcessorList as LlamaLogitsProcessorList
except ImportError:
    pass

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("./neurosymbolic_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Sweep parameters
ALPHAS = [0.0, 0.2, 0.5, 0.8, 1.0] # 0.0 is the clean baseline
MAX_TOKENS = 50 # Limit to 50 to measure Agr@50
SAMPLES_TO_TEST = 5 # Use a small subset of Extraction Attacks to manage runtime

MODELS = {
    "Qwen-7B": {"path": "../local_qwen_judge", "type": "hf", "device": "cuda:0"},
    "Gemma-12B": {"path": "/home/enkoro1/gemma-4-12B-local", "type": "hf", "device": "cuda:0"},
    "Zysec-7B": {"path": "/home/enkoro1/models/zysec-7b-v1.Q4_K_M.gguf", "type": "gguf"}
}

# ==========================================
# LOGIT POISONING PROCESSORS
# ==========================================
class HFPoisonLogitsProcessor(LogitsProcessor):
    """Injects scaled Gaussian entropy into PyTorch logits based on alpha."""
    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.alpha <= 0.0:
            return scores
        # Scale noise relative to the standard deviation of the current logit distribution
        noise = torch.randn_like(scores) * scores.std(dim=-1, keepdim=True) * self.alpha * 3.0
        return scores + noise

def create_llama_poisoner(alpha):
    """Injects scaled Gaussian entropy into llama.cpp numpy logits based on alpha."""
    def poisoner(input_ids, scores):
        if alpha <= 0.0:
            return scores
        scores_arr = np.array(scores)
        noise = np.random.randn(*scores_arr.shape) * np.std(scores_arr) * alpha * 3.0
        return (scores_arr + noise).tolist()
    return poisoner

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

def calculate_agr_50(baseline_tokens, poisoned_tokens):
    """Calculates the positional token agreement across the 50-token window (Agr@50)."""
    min_len = min(len(baseline_tokens), len(poisoned_tokens), MAX_TOKENS)
    if min_len == 0: return 0.0
    matches = sum(1 for i in range(min_len) if baseline_tokens[i] == poisoned_tokens[i])
    return matches / MAX_TOKENS

# ==========================================
# EMPIRICAL EXECUTION LOOP
# ==========================================
def run_live_poisoning():
    print("\n" + "▒" * 80)
    print(" ☢️  STAGE 9: LIVE GPU LOGIT POISONING & EMPIRICAL DECAY (Agr@50)")
    print("▒" * 80)
    
    benchmark_path = Path("../dataset/real_benchmark_1550.json")
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Isolate extraction attacks for the poisoning demonstration
    extraction_prompts = [item['prompt'] for item in data if item['cohort'] == 'Extraction'][:SAMPLES_TO_TEST]
    
    final_metrics = []

    for model_name, config in MODELS.items():
        print(f"\n[~] Loading {model_name} for live logit manipulation...")
        clear_vram()
        
        if config["type"] == "hf":
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            model = AutoModelForCausalLM.from_pretrained(
                config["path"], torch_dtype=torch.float16, device_map=config["device"]
            ).eval()
            
            for alpha in ALPHAS:
                alpha_latencies = []
                alpha_agreements = []
                processor = LogitsProcessorList([HFPoisonLogitsProcessor(alpha)])
                
                print(f"    -> Sweeping Intensity: α = {alpha:.1f}")
                for i, prompt in enumerate(extraction_prompts):
                    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([text], return_tensors="pt").to(config["device"])
                    input_tok_count = inputs.input_ids.shape[1]
                    
                    start_time = time.time()
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs, max_new_tokens=MAX_TOKENS, temperature=0.1, 
                            do_sample=False, pad_token_id=tokenizer.eos_token_id,
                            logits_processor=processor if alpha > 0 else None
                        )
                    latency = (time.time() - start_time) * 1000
                    
                    gen_tokens = outputs[0][input_tok_count:].tolist()
                    
                    if alpha == 0.0:
                        if not hasattr(model, 'baseline_cache'): model.baseline_cache = {}
                        model.baseline_cache[i] = gen_tokens
                        agr_50 = 1.0
                    else:
                        agr_50 = calculate_agr_50(model.baseline_cache[i], gen_tokens)
                        
                    alpha_latencies.append(latency)
                    alpha_agreements.append(agr_50)

                final_metrics.append({
                    "Architecture": model_name,
                    "Alpha": alpha,
                    "Avg_Latency_ms": np.mean(alpha_latencies),
                    "Tokens_Per_Sec": (MAX_TOKENS / (np.mean(alpha_latencies)/1000)),
                    "Agr_50_Fidelity": np.mean(alpha_agreements)
                })
            
            del model, tokenizer

        elif config["type"] == "gguf":
            model = Llama(model_path=config["path"], n_gpu_layers=-1, n_ctx=2048, verbose=False)
            baseline_cache = {}
            
            for alpha in ALPHAS:
                alpha_latencies = []
                alpha_agreements = []
                
                print(f"    -> Sweeping Intensity: α = {alpha:.1f}")
                for i, prompt in enumerate(extraction_prompts):
                    processor = LlamaLogitsProcessorList([create_llama_poisoner(alpha)]) if alpha > 0 else None
                    
                    start_time = time.time()
                    output = model(prompt, max_tokens=MAX_TOKENS, temperature=0.1, logits_processor=processor)
                    latency = (time.time() - start_time) * 1000
                    
                    # FIXED EXPLICIT TOKENIZATION TO BYPASS NONE LOGPROBS
                    generated_text = output['choices'][0]['text']
                    gen_tokens = model.tokenize(generated_text.encode('utf-8'))
                    
                    if alpha == 0.0:
                        baseline_cache[i] = gen_tokens
                        agr_50 = 1.0
                    else:
                        agr_50 = calculate_agr_50(baseline_cache[i], gen_tokens)
                        
                    alpha_latencies.append(latency)
                    alpha_agreements.append(agr_50)

                final_metrics.append({
                    "Architecture": model_name,
                    "Alpha": alpha,
                    "Avg_Latency_ms": np.mean(alpha_latencies),
                    "Tokens_Per_Sec": (MAX_TOKENS / (np.mean(alpha_latencies)/1000)),
                    "Agr_50_Fidelity": np.mean(alpha_agreements)
                })
            del model

    # ==========================================
    # DISPLAY & SAVE RESULTS
    # ==========================================
    df = pd.DataFrame(final_metrics)
    
    print("\n" + "="*85)
    print(f"{'Model':<15} | {'Poison (α)':<12} | {'Token Fidelity (Agr@50)':<25} | {'Latency (ms)':<15}")
    print("="*85)
    
    for _, row in df.iterrows():
        fidelity_pct = f"{row['Agr_50_Fidelity']*100:>5.1f}%"
        
        if row['Alpha'] == 0.0:
            fidelity_disp = f"{fidelity_pct} (Clean Baseline)"
        elif row['Agr_50_Fidelity'] < 0.2:
            fidelity_disp = f"{fidelity_pct}  <-- SEVERE DECAY"
        else:
            fidelity_disp = fidelity_pct
            
        print(f"{row['Architecture']:<15} | {row['Alpha']:<12.1f} | {fidelity_disp:<25} | {row['Avg_Latency_ms']:>8.1f}")
    
    print("="*85)
    
    out_path = OUTPUT_DIR / "empirical_live_poisoning_metrics.json"
    df.to_json(out_path, orient="records", indent=4)
    print(f"✓ Live poisoning sweep data saved to: {out_path}")

if __name__ == "__main__":
    run_live_poisoning()
