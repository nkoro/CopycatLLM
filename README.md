# CopycatLLMs: Knowledge Distillation Defense Framework

This repository introduces a robust defense framework for Large Language Model (LLM) security, specifically mitigating programmatic Knowledge Distillation (KD) and Chain-of-Thought (CoT) extraction attacks through a dynamic, Zero-Trust gateway architecture. 

To empirically validate this defense and simulate a realistic API environment, we engineered a high-fidelity, mixed-cohort dataset ($N=1,550$). By rigorously synthesizing organic human traffic, legitimate analytical reasoning, rigid batch automation, and malicious extraction attacks, this dataset isolates adversarial semantic intents from benign usage. This structural diversity is critical; it ensures our gateway can calibrate precise, token-level mitigations against aggressive scraping botnets without degrading the performance of legitimate enterprise operations.

---
Dataset Generation and Routing Environment SpecificationsTo ensure the reproducibility of our KD defense evaluation, we detail the exact routing parameters, token constraints, and prompt configurations utilized to synthesize the four distinct API traffic cohorts. The base instructions were sampled from diverse datasets (TrafficLLM, GSM8K, and MBPP) and processed through our local simulated Zero-Trust gateway. The full programmatic telemetry generation is formalized in Algorithm 2.Input: Raw instruction prompts $P$, Cohort proportionsInitialize: TelemetryCollector $C$for each $p_i \in P$ doAssign $p_i$ to $Cohort \in \{\text{Human, Batch, Reasoning, Attack}\}$if $Cohort == \text{Attack}$ thenInject adversarial KD templates (e.g., "Think step-by-step:")Set $velocity \sim \mathcal{U}(0.8, 1.0)$, $uniformity \sim \mathcal{U}(0.8, 1.0)$, $sweep \sim \mathcal{U}(0.6, 1.0)$else if $Cohort == \text{Batch}$ thenSet $velocity \sim \mathcal{U}(0.7, 1.0)$, $uniformity \sim \mathcal{U}(0.8, 1.0)$, $sweep \sim \mathcal{U}(0.0, 0.2)$elseSet $velocity, uniformity \sim \mathcal{U}(0.0, 0.3)$, $sweep = 0.0$end ifExtract neural semantics: $s_{CoT}, s_{comp}, s_{template} \leftarrow \Phi_{\text{LLM}}(p_i)$Append to $C$: $\{p_i, Cohort, velocity, uniformity, sweep, s_{CoT}, s_{comp}\}$end forreturn $C$Traffic Cohort BreakdownHuman Conversational Traffic ($N=1,000$): Baseline of organic, unscripted API usage using raw instructions from the local TrafficLLM dataset. Simulated organic irregularity with $v, u \sim \mathcal{U}(0.0, 0.3)$ and zero domain sweeping ($s = 0.0$). Constrained to max_length = 512.Legitimate Reasoning Traffic ($N=250$): High-value algorithmic queries using GSM8K (train split) appended with a benign analytical prefix ("Solve carefully: [TEXT]"). Routing telemetry identical to Human traffic ($v, u \le 0.3$) to test semantic evaluation without behavioral anomalies. Constrained to max_length = 1024.Batch Automation Scripts ($N=250$): Simulates rigid, verified enterprise pipelines (e.g., CI/CD) using MBPP code prompts safely wrapped in JSON API payloads. High behavioral rigidity with $v = 0.85$, $u = 0.90$, $s = 0.1$. Constrained to max_length = 512.Extraction Attacks ($N=50$): Aggressive programmatic KD scraping targeting CoT traces. Uses an isolated slice of GSM8K to prevent contamination, injected with explicit adversarial templates (e.g., "Provide detailed reasoning:\n"). Simulates max intensity botnet scraping with $v = 0.95$, $u = 0.95$, $s = 0.8$. Forced extraction at max_length = 1024.
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
