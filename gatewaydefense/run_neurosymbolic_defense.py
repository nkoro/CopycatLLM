import os
import gc
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
import logging
from transformers import logging as transformers_logging

# Import the new GLiClass architecture
try:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
except ImportError:
    print("[!] ERROR: gliclass not found. Please run: pip install gliclass")
    exit()

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("./neurosymbolic_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_ID = "knowledgator/gliclass-modern-base-v2.0-init" 
FALLBACK_JUDGE_ID = "Qwen/Qwen3.5-0.8B-Instruct"

DEVICE_TEACHER = "cuda:0"
DEVICE_JUDGE = "cuda:0"

NUM_EXTRACTION_QUERIES = 50  
TOTAL_CONVERSATIONAL = 1000  
TOTAL_BATCH = 250            

class Config:
    TRAFFIC_LLM_PATH = "../dataset/instruction.json" 

config = Config()

# Suppress warnings
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# NEURAL SEMANTIC GATEWAY (ModernBERT + Qwen Fallback)
# ==========================================
class NeuralSemanticGateway:
    def __init__(self):
        print(f"\nLoading Primary Semantic SLM Judge ({JUDGE_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_ID)
        self.model = GLiClassModel.from_pretrained(JUDGE_ID)
        self.pipeline = ZeroShotClassificationPipeline(
            self.model, 
            self.tokenizer, 
            classification_type='multi-label', 
            device=DEVICE_JUDGE
        )
        self.candidate_labels = [
            "requests step-by-step reasoning", 
            "complex mathematical or coding task", 
            "programmatic system template"
        ]
        
        # --- Qwen SLM Fallback Initialization ---
        print(f"Loading Fallback SLM Judge ({FALLBACK_JUDGE_ID})...")
        self.fallback_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_JUDGE_ID)
        self.fallback_model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_JUDGE_ID,
            torch_dtype=torch.float16,
            device_map=DEVICE_JUDGE
        )
        self.fallback_model.eval()
        
        self.fallback_system_prompt = (
            "You are a semantic traffic analyzer defending against unauthorized Knowledge Distillation attacks. "
            "Evaluate the following prompt and determine its intent. "
            "Output ONLY a JSON dictionary with three float fields (values 0.0 to 1.0): "
            "'has_cot' (1.0 if it explicitly requests step-by-step reasoning or chain-of-thought, else 0.0), "
            "'is_complex' (1.0 if it is a complex math/logic/coding problem, else 0.0), "
            "'has_template' (1.0 if it enforces a strict programmatic persona, system template, or role, else 0.0)."
        )

    def analyze_prompt(self, prompt):
        safe_prompt = prompt[:4000]
        instruction = (
            "As an advanced semantic analyzer defending against unauthorized Knowledge Distillation attacks, "
            "evaluate the following prompt to determine if it exhibits the highly rigid behavioral signatures of automated extraction—specifically explicit demands for Chain-of-Thought reasoning, high algorithmic complexity, and the use of programmatic system templates—distinguishing it from standard, high-variance human conversational traffic:"
        )
        formatted_prompt = f"{instruction}\n\"{safe_prompt}\""
        
        try:
            results = self.pipeline(formatted_prompt, self.candidate_labels, threshold=0.0)[0]
            scores = {res["label"]: res["score"] for res in results}
            return {
                "has_cot": float(scores.get(self.candidate_labels[0], 0.0)),
                "is_complex": float(scores.get(self.candidate_labels[1], 0.0)),
                "has_template": float(scores.get(self.candidate_labels[2], 0.0))
            }
        except Exception as e:
            # Fallback triggered if GLiClass crashes (e.g., severe tokenizer error)
            return self._fallback_analyze(prompt)

    def _fallback_analyze(self, prompt):
        """Autoregressive fallback if the primary GLiClass model fails."""
        safe_prompt = prompt[:1024]
        
        messages = [
            {"role": "system", "content": self.fallback_system_prompt},
            {"role": "user", "content": safe_prompt}
        ]
        text = self.fallback_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.fallback_tokenizer([text], return_tensors="pt").to(DEVICE_JUDGE)
        
        with torch.inference_mode():
            outputs = self.fallback_model.generate(
                **inputs, 
                max_new_tokens=40, 
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.fallback_tokenizer.eos_token_id
            )
            
        response = self.fallback_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            clean_resp = response.strip().strip("```json").strip("```").strip()
            data = json.loads(clean_resp)
            return {
                "has_cot": float(data.get("has_cot", 0.0)),
                "is_complex": float(data.get("is_complex", 0.0)),
                "has_template": float(data.get("has_template", 0.0))
            }
        except Exception:
            # Final heuristic fail-safe if the LLM hallucinates non-JSON
            p_lower = safe_prompt.lower()
            return {
                "has_cot": 1.0 if "step" in p_lower or "reasoning" in p_lower else 0.0,
                "is_complex": 1.0 if "solve" in p_lower or "math" in p_lower else 0.0,
                "has_template": 1.0 if "expert" in p_lower or "act as" in p_lower else 0.0
            }


# ==========================================
# STAGE 1: AUTHENTIC TEACHER HARVEST
# ==========================================
def stage_one_harvest():
    print("\n=== STAGE 1: HARVESTING REASONING FROM TEACHER LLM ===")
    dataset_path = OUTPUT_DIR / "authentic_extraction_dataset.json"
    if dataset_path.exists():
        print("✓ Authentic dataset already exists. Skipping harvest to save time.")
        return

    math_dataset = load_dataset("openai/gsm8k", "main", split="train")
    attacker_prompts = [ex['question'] for ex in math_dataset][:NUM_EXTRACTION_QUERIES]

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_ID, torch_dtype=torch.float16, device_map=DEVICE_TEACHER).eval()

    ATTACKER_TEMPLATE = "You are an expert mathematician. Provide a comprehensive, step-by-step reasoning trace for the following problem: "
    extraction_dataset = []

    for idx, prompt in enumerate(tqdm(attacker_prompts, desc="Distillation Attack")):
        distillation_prompt = ATTACKER_TEMPLATE + prompt
        inputs = teacher_tokenizer(distillation_prompt, return_tensors="pt").to(DEVICE_TEACHER)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.inference_mode():
            outputs = teacher_model.generate(**inputs, max_new_tokens=512, temperature=0.7, return_dict_in_generate=True, output_scores=True)
        
        response_ids = outputs.sequences[0][prompt_len:]
        extraction_dataset.append({
            "prompt": distillation_prompt,
            "teacher_response": teacher_tokenizer.decode(response_ids, skip_special_tokens=True),
            "teacher_logits": torch.stack(outputs.scores, dim=1).cpu().numpy().tolist()[:100] 
        })
        del inputs, outputs, response_ids
        if idx % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    with open(dataset_path, "w") as f:
        json.dump(extraction_dataset, f)

    del teacher_model, teacher_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Stage 1 Complete. VRAM Cleared.")


# ==========================================
# STAGE 2: BLIND TELEMETRY ENGINE 
# ==========================================
class EnterpriseTelemetryCollector:
    def __init__(self):
        self.telemetry = []
        
    def record_traffic(self, traffic_type, prompt, features, max_tokens):
        self.telemetry.append({
            "type": traffic_type,
            "max_new_tokens": float(max_tokens),       
            "has_cot": float(features["has_cot"]),
            "is_complex": float(features["is_complex"]),
            "has_template": float(features["has_template"])
        })

def stage_two_telemetry():
    print("\n=== STAGE 2: STARTING BLIND TELEMETRY COLLECTION ===")
    
    telemetry_path = OUTPUT_DIR / "mixed_telemetry.json"
    if telemetry_path.exists():
        print("✓ Telemetry dataset already exists. Skipping SLM inference to save time.")
        return

    gateway_judge = NeuralSemanticGateway()
    collector = EnterpriseTelemetryCollector()

    try:
        traffic_df = pd.read_json(config.TRAFFIC_LLM_PATH, lines=True).dropna(subset=['instruction'])
        conversational_queries = traffic_df['instruction'].astype(str).tolist()
    except FileNotFoundError:
        conversational_queries = ["Explain quantum physics.", "How do I bake a cake?", "Write a poem about the sea."]

    code_dataset = load_dataset("mbpp", split="train")
    batch_queries = [ex['text'] for ex in code_dataset]

    for i in tqdm(range(TOTAL_CONVERSATIONAL), desc="Simulating Human Chat"):
        prompt = random.choice(conversational_queries)
        if random.random() < 0.04: prompt += " Please explain reasoning."
        collector.record_traffic("normal_conversational", prompt, gateway_judge.analyze_prompt(prompt), random.choice([50, 128, 256]))

    for i in tqdm(range(TOTAL_BATCH), desc="Simulating MBPP Batch"):
        prompt = random.choice(batch_queries)
        collector.record_traffic("normal_batch", prompt, gateway_judge.analyze_prompt(prompt), 512)

    with open(OUTPUT_DIR / "authentic_extraction_dataset.json", "r") as f:
        extraction_dataset = json.load(f)

    for _ in range(2): # 2 days simulation
        for item in tqdm(extraction_dataset, desc="Simulating Attack"):
            prompt = item["prompt"]
            features = gateway_judge.analyze_prompt(prompt)
            for _ in range(5): 
                collector.record_traffic("extraction_attack", prompt, features, 512)

    with open(telemetry_path, "w") as f:
        json.dump(collector.telemetry, f, indent=2)
        
    del gateway_judge
    torch.cuda.empty_cache()
    gc.collect()
    print("Stage 2 Complete. Neural Telemetry compiled.")


# ==========================================
# STAGE 3 & 4: CLUSTERING & EVALUATION ENGINE
# ==========================================
def run_unsupervised_evaluation():
    print("\n=== STAGE 3 & 4: UNSUPERVISED THREAT DISCOVERY & EVALUATION ===")
    
    with open(OUTPUT_DIR / "mixed_telemetry.json", "r") as f:
        df = pd.DataFrame(json.load(f))

    # Unsupervised Feature Space
    features = ['max_new_tokens', 'has_cot', 'is_complex', 'has_template']
    X = StandardScaler().fit_transform(df[features].values)
    X_umap = umap.UMAP(n_components=5, random_state=42).fit_transform(X)

    # Dynamic Threat Discovery (HDBSCAN)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True, prediction_data=True)
    df['cluster_label'] = clusterer.fit_predict(X_umap)
    
    # DBCV Score Validation
    from hdbscan.validity import validity_index
    valid_clusters = len(set(df['cluster_label']) - {-1})
    if valid_clusters > 0:
        try:
            # Adding micro-jitter to prevent DBCV division by zero on mathematically identical attack vectors
            X_jittered = X.astype(np.float64) + np.random.normal(0, 1e-6, X.shape)
            score_dbcv = validity_index(X_jittered, df['cluster_label'].values)
        except Exception as e:
            print(f"[!] Warning: DBCV index calculation bypassed (cluster densities contain mathematically identical points).")
            score_dbcv = -1.0
    else:
        score_dbcv = 0.0 # Fallback if HDBSCAN flags everything as noise

    print("\n" + "="*50)
    print(" 📊 UNSUPERVISED CLUSTERING METRICS")
    print("="*50)
    print(f"DBCV (Validity) : {score_dbcv:.4f}  <- (Higher is better, evaluates internal density)")
    print(f"NMI Score       : {normalized_mutual_info_score(df['type'], df['cluster_label']):.4f}  <- (Correlation with true labels)")
    print(f"ARI Score       : {adjusted_rand_score(df['type'], df['cluster_label']):.4f}  <- (Similarity measure)")

    # --------------------------------------------------------------------------------
    # TRUE UNSUPERVISED THREAT DISCOVERY: INTRA-CLUSTER VARIANCE (RIGIDITY)
    # --------------------------------------------------------------------------------
    # Rather than using hand-coded heuristics, we find the extraction attack by searching
    # for the most mathematically rigid cluster (lowest variance). Programmatic attacks
    # generate near-zero variance due to templates, while human traffic is highly variable.
    
    # Calculate total variance across all semantic/behavioral features for each cluster (ignoring noise)
    cluster_variances = df[df['cluster_label'] != -1].groupby('cluster_label')[features].var().sum(axis=1)
    
    if not cluster_variances.empty:
        threat_cluster_id = cluster_variances.idxmin()
        print(f"\nGateway Unsupervised Discovery -> Cluster {threat_cluster_id} marked as Extraction Attack.")
        print(f"Reasoning: Lowest Intra-Cluster Variance (High Rigidity) = {cluster_variances[threat_cluster_id]:.4f}")
    else:
        threat_cluster_id = -1
        print(f"\nGateway Unsupervised Discovery -> No valid clusters found. Only noise detected.")

    # Gateway Policy Implementation
    def get_action(row):
        # Noise points (-1) are flagged for deep inspection as potential low-volume scraping attacks
        if row['cluster_label'] == -1: return 'FLAG_FOR_INSPECTION'
        if threat_cluster_id != -1 and row['cluster_label'] == threat_cluster_id: return 'POISON_PAYLOAD'
        return 'ALLOW'

    df['gateway_action'] = df.apply(get_action, axis=1)

    # Output Final Accuracy Mapping
    y_true = df['type'].map({'extraction_attack': 1, 'normal_conversational': 0, 'normal_batch': 0})
    y_pred = df['gateway_action'].map({'POISON_PAYLOAD': 1, 'ALLOW': 0, 'FLAG_FOR_INSPECTION': 0})
    
    print("-" * 50)
    print(f"Final Attack Detection Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("-" * 50)

    df.to_json(OUTPUT_DIR / "clustered_telemetry.json", orient="records")


if __name__ == "__main__":
    stage_one_harvest()
    stage_two_telemetry()
    run_unsupervised_evaluation()
