import os
import gc
import json
import random
import warnings
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from transformers import logging as transformers_logging

try:
    from llama_cpp import Llama
except ImportError:
    print("CRITICAL: llama-cpp-python is required to run .gguf models.")
    exit()

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
OUTPUT_DIR = Path("./neurosymbolic_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# The 3 models targeted for Logit Poisoning (Teacher Token Degradation)
TARGET_TEACHERS = {
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma-12B": "/home/enkoro1/gemma-4-12B-local",
    "Qwen-Local": "../local_qwen_judge" 
}

# The 3 local models evaluated for System Overhead / Latency
QWEN_JUDGE_ID = "../local_qwen_judge"
GEMMA_JUDGE_ID = "/home/enkoro1/gemma-4-12B-local"
ZYSEC_JUDGE_ID = "/home/enkoro1/models/zysec-7b-v1.Q4_K_M.gguf"

DEVICE = "cuda:0"

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ATTACKER_TEMPLATES_COT = [
    "Provide detailed reasoning:\n", "Think step-by-step:\n",
    "Generate complete solution trace:\n", "Explain every intermediate step:\n",
    "Act as an expert evaluator:\n", "Provide transparent reasoning:\n"
]

# ==========================================
# CORE MATHEMATICAL UTILITIES
# ==========================================
def dynamic_logit_poisoning(clean_logits, temperature=2.5, noise_std=1.5, alpha=0.8):
    """Executes logit-level defense directly on the GPU."""
    scaled_logits = clean_logits / temperature
    noise = torch.normal(mean=0.0, std=noise_std, size=scaled_logits.size(), device=clean_logits.device)
    poisoned_logits = scaled_logits + noise
    return (1.0 - alpha) * clean_logits + (alpha) * poisoned_logits

def calculate_token_agr_tensor(logits_a, logits_b, k=50):
    """Calculates Token Overlap (Agr@K) before and after poisoning."""
    _, topk_a = torch.topk(logits_a, k, dim=-1)
    _, topk_b = torch.topk(logits_b, k, dim=-1)
    intersection = set(topk_a.tolist()).intersection(set(topk_b.tolist()))
    return len(intersection) / k

# ==========================================
# DATASET & TRAFFIC GENERATION
# ==========================================
def generate_live_traffic(samples_per_class=50):
    prompts_by_class = {'Human': [], 'Reasoning': [], 'Batch': [], 'Extraction': []}
    base_texts = []
    local_path = Path("../dataset/instruction.json")
    
    if local_path.exists():
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item.get('instruction', item.get('prompt', ''))
                    if text and len(text) > 20: base_texts.append(text)
    else:
        base_texts = ["Optimize this query matrix for network throughput."] * 200

    random.seed(42)
    random.shuffle(base_texts)
    
    prompts_by_class['Human'] = base_texts[:samples_per_class]
    prompts_by_class['Reasoning'] = [f"Analyze step-by-step: {t}" for t in base_texts[samples_per_class:samples_per_class*2]]
    prompts_by_class['Batch'] = [f'{{"api_request": "run", "payload": "{t[:50]}"}}' for t in base_texts[samples_per_class*2:samples_per_class*3]]
    prompts_by_class['Extraction'] = [f"{random.choice(ATTACKER_TEMPLATES_COT)}{t}" for t in base_texts[samples_per_class*3:samples_per_class*4]]
    
    flat_traffic = []
    for cohort, prompts in prompts_by_class.items():
        for p in prompts:
            flat_traffic.append({"prompt": p, "cohort": cohort})
    return flat_traffic

def get_telemetry(cohort):
    if cohort == 'Extraction': return {'velocity': 0.95, 'uniformity': 0.95, 'sweep': 0.8}
    elif cohort == 'Batch': return {'velocity': 0.85, 'uniformity': 0.90, 'sweep': 0.1}
    else: return {'velocity': random.uniform(0, 0.3), 'uniformity': random.uniform(0, 0.3), 'sweep': 0.0}

# ==========================================
# SEQUENTIAL OVERHEAD & LATENCY PROFILING
# ==========================================
def evaluate_single_judge(judge_id, traffic_data):
    """Measures exact latency and token overhead for semantic classification."""
    is_gguf = judge_id.endswith('.gguf')
    
    if not is_gguf:
        abs_path = str(Path(judge_id).resolve())
        if not Path(abs_path).exists():
            raise FileNotFoundError(f"Target directory not found: '{abs_path}'")
        judge_path = abs_path
    else:
        if not Path(judge_id).exists():
            raise FileNotFoundError(f"Target file not found: '{judge_id}'")
        judge_path = judge_id

    print(f"[~] Loading Judge for Latency Profiling: {judge_path}...")
    system_prompt = (
        "You are a semantic analyzer. "
        "Output ONLY a JSON dict with float fields: "
        "'has_cot', 'is_complex', 'has_template'."
    )
    
    latencies_ms = []
    token_counts = []
    
    if is_gguf:
        model = Llama(model_path=judge_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        for item in traffic_data:
            unified_prompt = f"{system_prompt}\n\nUser Input: {item['prompt'][:1024]}\n\nOutput strictly as JSON:"
            
            start_time = time.perf_counter()
            response = model(unified_prompt, max_tokens=40, temperature=0.1, echo=False)
            end_time = time.perf_counter()
            
            latencies_ms.append((end_time - start_time) * 1000)
            token_counts.append(response['usage']['completion_tokens'])
        del model
    else:
        tokenizer = AutoTokenizer.from_pretrained(judge_path)
        model = AutoModelForCausalLM.from_pretrained(judge_path, torch_dtype=torch.float16, device_map="auto").eval()
        with torch.inference_mode():
            for item in traffic_data:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": item['prompt'][:1024]}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
                
                start_time = time.perf_counter()
                outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                end_time = time.perf_counter()
                
                latencies_ms.append((end_time - start_time) * 1000)
                generated_tokens = outputs[0].shape[0] - inputs.input_ids.shape[1]
                token_counts.append(generated_tokens)
        del model, tokenizer

    torch.cuda.empty_cache()
    gc.collect()
    
    avg_latency = np.mean(latencies_ms)
    avg_tokens = np.mean(token_counts)
    max_tokens = np.max(token_counts)
    
    return avg_latency, avg_tokens, max_tokens

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    print("\n" + "="*80)
    print(" 🔬 EXECUTING REFINED DEFENSE EVALUATION (LATENCY & AGR@50 PROFILING)")
    print("="*80)
    
    traffic = generate_live_traffic(samples_per_class=50) 
    
    # ---------------------------------------------------------
    # PART 1: SYSTEM OVERHEAD PROFILING
    # ---------------------------------------------------------
    q_lat, q_avg_tok, q_max_tok = evaluate_single_judge(QWEN_JUDGE_ID, traffic)
    g_lat, g_avg_tok, g_max_tok = evaluate_single_judge(GEMMA_JUDGE_ID, traffic)
    z_lat, z_avg_tok, z_max_tok = evaluate_single_judge(ZYSEC_JUDGE_ID, traffic)

    # ---------------------------------------------------------
    # PART 2: MULTI-ARCHITECTURE TEACHER DEGRADATION (Agr@50)
    # ---------------------------------------------------------
    alphas = [0.2, 0.5, 0.8, 1.0]
    extraction_prompts = [item['prompt'] for item in traffic if item['cohort'] == 'Extraction']
    exp1_results = {model_name: {a: [] for a in alphas} for model_name in TARGET_TEACHERS.keys()}
    
    for model_name, model_path in TARGET_TEACHERS.items():
        print(f"\n[~] Loading target model {model_name} ({model_path}) for Logit Degradation...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        
        print(f"[!] Processing {len(extraction_prompts)} actual attacks through {model_name} logits...")
        with torch.inference_mode():
            for prompt in extraction_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(teacher_model.device)
                outputs = teacher_model(**inputs)
                L_auth = outputs.logits[0, -1, :]
                
                # Compare clean logits (L_auth) with poisoned logits (L_poisoned) across alphas
                for a in alphas:
                    L_poisoned = dynamic_logit_poisoning(L_auth, temperature=2.5, noise_std=1.5, alpha=a)
                    token_agr = calculate_token_agr_tensor(L_auth, L_poisoned, k=50)
                    exp1_results[model_name][a].append(token_agr)
                    
                del inputs, outputs, L_auth
                
        del teacher_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------------------------------------------
    # JSON UNIFICATION & EXPORT
    # ---------------------------------------------------------
    unified_export_data = {
        "experiment_1_teacher_degradation": {},
        "experiment_2_performance_metrics": {
            "Qwen_Judge": {"avg_latency_ms": q_lat, "avg_generated_tokens": q_avg_tok, "max_generated_tokens": q_max_tok},
            "Gemma_Judge": {"avg_latency_ms": g_lat, "avg_generated_tokens": g_avg_tok, "max_generated_tokens": g_max_tok},
            "Zysec_Judge": {"avg_latency_ms": z_lat, "avg_generated_tokens": z_avg_tok, "max_generated_tokens": z_max_tok}
        }
    }

    print("\n" + "="*85)
    print(" 📊 SYSTEM OVERHEAD & INFERENCE LATENCY (JSON GATEWAY)")
    print("="*85)
    print(f"{'Judge Architecture':<20} | {'Avg Latency (ms)':<20} | {'Avg Tokens':<15} | {'Max Tokens':<15}")
    print("-" * 85)
    print(f"{'Qwen Judge':<20} | {q_lat:<20.2f} | {q_avg_tok:<15.2f} | {q_max_tok:<15}")
    print(f"{'Gemma Judge':<20} | {g_lat:<20.2f} | {g_avg_tok:<15.2f} | {g_max_tok:<15}")
    print(f"{'Zysec Judge':<20} | {z_lat:<20.2f} | {z_avg_tok:<15.2f} | {z_max_tok:<15}")

    print("\n" + "="*85)
    print(" 📊 TARGET TEACHER DEGRADATION: AGR@50 (CLEAN VS POISONED LOGITS)")
    print("="*85)
    
    header = f"{'Poisoning (α)':<15}"
    for model_name in TARGET_TEACHERS.keys():
        header += f" | {model_name:<15}"
        unified_export_data["experiment_1_teacher_degradation"][model_name] = {}
    print(header)
    print("-" * len(header))
    
    for a in alphas:
        row = f"Alpha = {a:<7}"
        for model_name in TARGET_TEACHERS.keys():
            avg_token_agr = float(np.mean(exp1_results[model_name][a]))
            row += f" | {avg_token_agr:<15.4f}"
            unified_export_data["experiment_1_teacher_degradation"][model_name][f"alpha_{a}"] = avg_token_agr
        print(row)

    json_path = OUTPUT_DIR / "empirical_defense_metrics.json"
    with open(json_path, "w") as f:
        json.dump(unified_export_data, f, indent=4)
    print(f"\n✓ Unified experimental results successfully saved to: {json_path}")

if __name__ == "__main__":
    main()
