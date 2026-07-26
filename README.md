# CopycatLLMs: Knowledge Distillation Defense Framework

This repository introduces a robust defense framework designed to mitigate programmatic Knowledge Distillation (KD) and Chain-of-Thought (CoT) extraction attacks via a dynamic Zero-Trust gateway. 

To empirically validate this defense, we engineered a high-fidelity, mixed-cohort dataset (N=1,550) that isolates malicious scraping intents from benign enterprise usage, allowing our gateway to calibrate precise token-level mitigations without degrading legitimate operations.

## Dataset Generation

To ensure the reproducibility of our KD defense evaluation, we detail the exact routing parameters, token constraints, and prompt configurations utilized to synthesize the four distinct API traffic cohorts. The base instructions were sampled from diverse datasets (TrafficLLM, GSM8K, and MBPP) and processed through our local simulated Zero-Trust gateway. The full programmatic telemetry generation is formalized in Algorithm 2.

---
**Algorithm 2** KD Attack and Telemetry Dataset Generation
---
1: **Input:** Raw instruction prompts $P$, Cohort proportions  
2: **Initialize:** *TelemetryCollector* $C$  
3: **for each** $p_i \in P$ **do**  
4: &nbsp;&nbsp;&nbsp;&nbsp;Assign $p_i$ to $Cohort \in \{\text{Human, Batch, Reasoning, Attack}\}$  
5: &nbsp;&nbsp;&nbsp;&nbsp;**if** $Cohort == \text{Attack}$ **then**  
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inject adversarial KD templates (e.g., "Think step-by-step:")  
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $velocity \sim \mathcal{U}(0.8, 1.0)$, $uniformity \sim \mathcal{U}(0.8, 1.0)$, $sweep \sim \mathcal{U}(0.6, 1.0)$  
8: &nbsp;&nbsp;&nbsp;&nbsp;**else if** $Cohort == \text{Batch}$ **then**  
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $velocity \sim \mathcal{U}(0.7, 1.0)$, $uniformity \sim \mathcal{U}(0.8, 1.0)$, $sweep \sim \mathcal{U}(0.0, 0.2)$  
10: &nbsp;&nbsp;&nbsp;&nbsp;**else**  
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $velocity, uniformity \sim \mathcal{U}(0.0, 0.3)$, $sweep = 0.0$  
12: &nbsp;&nbsp;&nbsp;&nbsp;**end if**  
13: &nbsp;&nbsp;&nbsp;&nbsp;Extract neural semantics: $s_{CoT}, s_{comp}, s_{template} \leftarrow \Phi_{\text{LLM}}(p_i)$  
14: &nbsp;&nbsp;&nbsp;&nbsp;Append to $C$: $\{p_i, Cohort, velocity, uniformity, sweep, s_{CoT}, s_{comp}\}$  
15: **end for**  
16: **return** $C$  

---

### Traffic Cohort Breakdown

* **Human Conversational Traffic (N=1,000):** Baseline of organic, unscripted API usage using raw instructions from the local TrafficLLM dataset. Simulated organic irregularity with $v, u \sim \mathcal{U}(0.0, 0.3)$ and zero domain sweeping ($s = 0.0$). Constrained to **max_length = 512**.
* **Legitimate Reasoning Traffic (N=250):** High-value algorithmic queries using GSM8K (train split) appended with a benign analytical prefix ("Solve carefully: [TEXT]"). Routing telemetry identical to Human traffic ($v, u \le 0.3$) to test semantic evaluation without behavioral anomalies. Constrained to **max_length = 1024**.
* **Batch Automation Scripts (N=250):** Simulates rigid, verified enterprise pipelines (e.g., CI/CD) using MBPP code prompts safely wrapped in JSON API payloads. High behavioral rigidity with $v = 0.85$, $u = 0.90$, $s = 0.1$. Constrained to **max_length = 512**.
* **Extraction Attacks (N=50):** Aggressive programmatic KD scraping targeting CoT traces. Uses an isolated slice of GSM8K to prevent contamination, injected with explicit adversarial templates (e.g., "Provide detailed reasoning:\n"). Simulates max intensity botnet scraping with $v = 0.95$, $u = 0.95$, $s = 0.8$. Forced extraction at **max_length = 1024**.
