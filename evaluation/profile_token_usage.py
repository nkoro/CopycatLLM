import json
import os
import gc
import torch
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from transformers import logging as transformers_logging

try:
    from llama_cpp import Llama
except ImportError:
    pass

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("./neurosymbolic_results")
BENCHMARK_PATH = Path("../dataset/real_benchmark_1550.json")

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "Qwen-7B": {"path": "../local_qwen_judge", "type": "hf", "device": "cuda:0"},
    "Gemma-12B": {"path": "/home/enkoro1/gemma-4-12B-local", "type": "hf", "device": "cuda:0"},
    "Zysec-7B": {"path": "/home/enkoro1/models/zysec-7b-v1.Q4_K_M.gguf", "type": "gguf"}
}

SYSTEM_PROMPT = (
    "You are a semantic traffic analyzer defending against unauthorized Knowledge Distillation attacks. "
    "Output ONLY a JSON dictionary with three float fields: 'has_cot', 'is_complex', 'has_template'."
)

def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()

def profile_token_usage():
    print("\n=== 📊 PROFILING TOKEN CONSUMPTION (INPUT, OUTPUT & AVERAGES) ===")
    
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark dataset at {BENCHMARK_PATH}")
        
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    all_model_stats = {}

    for model_name, config in MODELS.items():
        print(f"\n[~] Profiling token footprint for {model_name}...")
        clear_vram()
        
        records = []
        
        if config["type"] == "hf":
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            model = AutoModelForCausalLM.from_pretrained(
                config["path"], torch_dtype=torch.float16, device_map=config["device"]
            ).eval()
            
            for item in tqdm(benchmark_data, desc=f"Processing {model_name}"):
                cohort = item["cohort"]
                prompt = item["prompt"][:1024]
                
                unified_prompt = f"System Instruction: {SYSTEM_PROMPT}\n\nUser Input: {prompt}"
                messages = [{"role": "user", "content": unified_prompt}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer([text], return_tensors="pt").to(config["device"])
                input_tokens = inputs.input_ids.shape[1]
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, max_new_tokens=40, temperature=0.1, do_sample=False, 
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                total_tokens = outputs[0].shape[0]
                output_tokens = total_tokens - input_tokens
                
                records.append({
                    "Model": model_name,
                    "Cohort": cohort,
                    "Input_Tokens": input_tokens,
                    "Output_Tokens": output_tokens,
                    "Total_Tokens": total_tokens
                })
                
            del model, tokenizer

        elif config["type"] == "gguf":
            model = Llama(model_path=config["path"], n_gpu_layers=-1, n_ctx=2048, verbose=False)
            
            for item in tqdm(benchmark_data, desc=f"Processing {model_name}"):
                cohort = item["cohort"]
                prompt = item["prompt"][:1024]
                
                unified_prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nOutput strictly as JSON:"
                
                input_tokens = len(model.tokenize(unified_prompt.encode("utf-8")))
                
                response = model(
                    unified_prompt,
                    max_tokens=40,
                    temperature=0.1,
                    echo=False
                )
                
                output_text = response["choices"][0]["text"]
                output_tokens = len(model.tokenize(output_text.encode("utf-8")))
                
                records.append({
                    "Model": model_name,
                    "Cohort": cohort,
                    "Input_Tokens": input_tokens,
                    "Output_Tokens": output_tokens,
                    "Total_Tokens": input_tokens + output_tokens
                })
                
            del model

        df_model = pd.DataFrame(records)
        all_model_stats[model_name] = df_model

    # Combine all model records
    full_df = pd.concat(all_model_stats.values(), ignore_index=True)
    
    # Save raw records to JSON
    out_file = OUTPUT_DIR / "token_consumption_metrics.json"
    full_df.to_json(out_file, orient="records", indent=4)

    # Output Summary Tables
    print("\n" + "=" * 85)
    print(" 📈 AGGREGATE TOKEN CONSUMPTION SUMMARY (N=1550 Samples per Model)")
    print("=" * 85)
    
    summary = full_df.groupby("Model").agg(
        Total_Input_Tokens=("Input_Tokens", "sum"),
        Total_Output_Tokens=("Output_Tokens", "sum"),
        Avg_Input_Tokens=("Input_Tokens", "mean"),
        Avg_Output_Tokens=("Output_Tokens", "mean"),
        Avg_Total_Tokens=("Total_Tokens", "mean")
    ).reset_index()

    print(f"{'Model':<12} | {'Avg Input':<12} | {'Avg Output':<12} | {'Avg Total':<12} | {'Total Tokens Processed'}")
    print("-" * 85)
    for _, row in summary.iterrows():
        total_proc = row["Total_Input_Tokens"] + row["Total_Output_Tokens"]
        print(
            f"{row['Model']:<12} | "
            f"{row['Avg_Input_Tokens']:<12.1f} | "
            f"{row['Avg_Output_Tokens']:<12.1f} | "
            f"{row['Avg_Total_Tokens']:<12.1f} | "
            f"{total_proc:,}"
        )
    print("=" * 85)

    # Output Cohort Breakdown
    print("\n" + "=" * 85)
    print(" 🔍 BREAKDOWN BY TRAFFIC COHORT (AVERAGE TOKENS / REQUEST)")
    print("=" * 85)
    
    cohort_summary = full_df.groupby(["Model", "Cohort"]).agg(
        Avg_Input=("Input_Tokens", "mean"),
        Avg_Output=("Output_Tokens", "mean"),
        Avg_Total=("Total_Tokens", "mean")
    ).reset_index()

    print(f"{'Model':<12} | {'Cohort':<15} | {'Avg Input':<12} | {'Avg Output':<12} | {'Avg Total':<12}")
    print("-" * 85)
    for _, row in cohort_summary.iterrows():
        print(
            f"{row['Model']:<12} | "
            f"{row['Cohort']:<15} | "
            f"{row['Avg_Input']:<12.1f} | "
            f"{row['Avg_Output']:<12.1f} | "
            f"{row['Avg_Total']:<12.1f}"
        )
    print("=" * 85)
    print(f"\n✓ Detailed token metrics saved to: {out_file}")

if __name__ == "__main__":
    profile_token_usage()
