# CopycatLLMs: Knowledge Distillation Defense Framework

This repository introduces a robust defense framework for Large Language Model (LLM) security, specifically mitigating programmatic Knowledge Distillation (KD) and Chain-of-Thought (CoT) extraction attacks through a dynamic, Zero-Trust gateway architecture. 

To empirically validate this defense and simulate a realistic API environment, we engineered a high-fidelity, mixed-cohort dataset ($N=1,550$). By rigorously synthesizing organic human traffic, legitimate analytical reasoning, rigid batch automation, and malicious extraction attacks, this dataset isolates adversarial semantic intents from benign usage. This structural diversity is critical; it ensures our gateway can calibrate precise, token-level mitigations against aggressive scraping botnets without degrading the performance of legitimate enterprise operations.

---

## Dataset Generation and Routing Environment Specifications

To ensure the reproducibility of our Knowledge Distillation (KD) defense evaluation, we detail the exact routing parameters, token constraints, and prompt configurations utilized to synthesize the four distinct API traffic cohorts. The base instructions were sampled from diverse datasets (TrafficLLM, GSM8K, and MBPP) and processed through our local simulated Zero-Trust gateway. The full programmatic telemetry generation is formalized in Algorithm 1.

```text
Algorithm 1: KD Attack and Telemetry Dataset Generation
---------------------------------------------------------------------------------
Input: Raw instruction sources P_TrafficLLM, P_GSM8K, P_MBPP
Initialize: TelemetryCollector C

For each cohort allocation:
    If Cohort == Human:
        Sample p_i ∈ P_TrafficLLM
        Set velocity, uniformity ~ U(0.0, 0.3), sweep = 0.0
        
    Else If Cohort == Reasoning:
        Sample p_i ∈ P_GSM8K (Index 0–249)
        Prepend benign template: "Solve carefully: "
        Set velocity, uniformity ~ U(0.0, 0.3), sweep = 0.0
        
    Else If Cohort == Batch:
        Sample p_i ∈ P_MBPP
        Sanitize internal quotes in p_i to single quotes
        Wrap p_i in JSON API schema with integer index
        Set velocity ~ U(0.7, 1.0), uniformity ~ U(0.8, 1.0), sweep ~ U(0.0, 0.2)
        
    Else If Cohort == Attack:
        Sample p_i ∈ P_GSM8K (Index 250–299 to prevent contamination)
        Inject random adversarial KD template (e.g., "Think step-by-step:\n")
        Set velocity ~ U(0.8, 1.0), uniformity ~ U(0.8, 1.0), sweep ~ U(0.6, 1.0)

    Extract neural semantics: s_CoT, s_comp, s_template ← Φ_LLM(p_i)
    Append to C: {p_i, Cohort, velocity, uniformity, sweep, s_CoT, s_comp}

Return C
