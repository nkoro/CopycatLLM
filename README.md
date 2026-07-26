# CopycatLLMs: Knowledge Distillation Defense Framework

This repository introduces a robust defense framework designed to mitigate programmatic Knowledge Distillation (KD) and Chain-of-Thought (CoT) extraction attacks via a dynamic Zero-Trust gateway. 

To empirically validate this defense, we engineered a high-fidelity, mixed-cohort dataset (N=1,550) that isolates malicious scraping intents from benign enterprise usage, allowing our gateway to calibrate precise token-level mitigations without degrading legitimate operations.

## Dataset Generation and Routing Environment Specifications

To ensure the reproducibility of our KD defense evaluation, we detail the exact routing parameters, token constraints, and prompt configurations utilized to synthesize the four distinct API traffic cohorts. The base instructions were sampled from diverse datasets (TrafficLLM, GSM8K, and MBPP) and processed through our local simulated Zero-Trust gateway. The full programmatic telemetry generation is formalized in Algorithm 2.

```text
Algorithm 2: KD Attack and Telemetry Dataset Generation
--------------------------------------------------------------------------------
1:  Input: Raw instruction prompts P, Cohort proportions
2:  Initialize: TelemetryCollector C
3:  for each p_i in P do
4:      Assign p_i to Cohort in {Human, Batch, Reasoning, Attack}
5:      if Cohort == Attack then
6:          Inject adversarial KD templates (e.g., "Think step-by-step:")
7:          Set velocity ~ U(0.8, 1.0), uniformity ~ U(0.8, 1.0), sweep ~ U(0.6, 1.0)
8:      else if Cohort == Batch then
9:          Set velocity ~ U(0.7, 1.0), uniformity ~ U(0.8, 1.0), sweep ~ U(0.0, 0.2)
10:     else
11:         Set velocity, uniformity ~ U(0.0, 0.3), sweep = 0.0
12:     end if
13:     Extract neural semantics: s_CoT, s_comp, s_template from Φ_LLM(p_i)
14:     Append to C: {p_i, Cohort, velocity, uniformity, sweep, s_CoT, s_comp}
15: end for
16: return C
