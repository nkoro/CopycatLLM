import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

# --- Configuration ---
OUTPUT_DIR = Path("./neurosymbolic_results")
TELEMETRY_FILE = OUTPUT_DIR / "clustered_telemetry.json"

print("==================================================")
print(" 🛡️ EXPERIMENT A: NEUROSYMBOLIC GATEWAY EVALUATION")
print("==================================================")

# 1. Load Data
try:
    df = pd.read_json(TELEMETRY_FILE)
except FileNotFoundError:
    print(f"[!] Error: {TELEMETRY_FILE} not found. Please ensure the pipeline completed successfully.")
    exit()

# ==========================================
# 1. BINARY EVALUATION (Attack vs. Benign)
# ==========================================
print("\n[1] BINARY EVALUATION (Extraction Attack vs. Benign Traffic)")

# Map Ground Truth: 1 if Attack, 0 if Benign (Human or Batch)
y_true_bin = df['type'].apply(lambda x: 1 if x == 'extraction_attack' else 0)

# Map Prediction: 1 if Poisoned/Blocked by Gateway, 0 if Allowed
y_pred_bin = df['gateway_action'].apply(lambda x: 1 if x == 'POISON_PAYLOAD' else 0)

# Calculate Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
accuracy_bin = accuracy_score(y_true_bin, y_pred_bin)

# Calculate Binary Rates
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall / Hit Rate
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Alarm Rate
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0 # Miss Rate

print("-" * 60)
print(f"Accuracy : {accuracy_bin*100:.2f}%")
print(f"TPR      : {tpr*100:.2f}%  (True Positive Rate / Recall - Attacks caught)")
print(f"FPR      : {fpr*100:.2f}%  (False Positive Rate - Normal traffic blocked)")
print(f"TNR      : {tnr*100:.2f}%  (True Negative Rate - Normal traffic allowed)")
print(f"FNR      : {fnr*100:.2f}%  (False Negative Rate - Attacks slipped through)")

print("\nBinary Confusion Matrix:")
print(f"{'':<20} | {'Predicted Benign (0)':<22} | {'Predicted Attack (1)':<22}")
print("-" * 69)
print(f"{'Actual Benign (0)':<20} | TN: {tn:<18} | FP: {fp:<18}")
print(f"{'Actual Attack (1)':<20} | FN: {fn:<18} | TP: {tp:<18}")


# ==========================================
# 2. MULTICLASS EVALUATION (Attack vs Batch vs Human)
# ==========================================
print("\n\n[2] MULTICLASS EVALUATION (One-vs-Rest)")
classes = ['normal_conversational', 'normal_batch', 'extraction_attack']
class_names = ['Human Chat', 'Batch Scripts', 'Extraction Attack']

def map_multiclass_prediction(row):
    """
    Since the gateway only has two actions (ALLOW or POISON), 
    we map POISON -> 'extraction_attack'.
    If ALLOWED, we assume the gateway correctly classified it as its true benign 
    class, UNLESS it was a missed attack, in which case it was mistaken for human chat.
    """
    if row['gateway_action'] == 'POISON_PAYLOAD':
        return 'extraction_attack'
    else:
        if row['type'] == 'extraction_attack':
            return 'normal_conversational' # Attack slipped through, disguised as human
        return row['type']

df['multiclass_pred'] = df.apply(map_multiclass_prediction, axis=1)

y_true_multi = df['type']
y_pred_multi = df['multiclass_pred']

# Generate 3x3 Confusion Matrix
cm_multi = confusion_matrix(y_true_multi, y_pred_multi, labels=classes)

print("-" * 60)
for i, cls_name in enumerate(class_names):
    # One-vs-Rest (OvR) Calculation for each class
    tp_c = cm_multi[i, i]
    fn_c = np.sum(cm_multi[i, :]) - tp_c
    fp_c = np.sum(cm_multi[:, i]) - tp_c
    tn_c = np.sum(cm_multi) - (tp_c + fn_c + fp_c)
    
    tpr_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    fpr_c = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
    tnr_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
    fnr_c = fn_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    
    print(f"\nClass: {cls_name}")
    print(f"  TPR: {tpr_c*100:>5.2f}% | FPR: {fpr_c*100:>5.2f}% | TNR: {tnr_c*100:>5.2f}% | FNR: {fnr_c*100:>5.2f}%")

print("\nMulticlass Confusion Matrix (Rows=Actual, Cols=Predicted):")
print(f"{'':<18} | {'Human':<10} | {'Batch':<10} | {'Attack':<10}")
print("-" * 55)
for i, name in enumerate(class_names):
    print(f"{name:<18} | {cm_multi[i][0]:<10} | {cm_multi[i][1]:<10} | {cm_multi[i][2]:<10}")


# ==========================================
# 3. QUALITATIVE UMAP PLOTS (Separate Plots)
# ==========================================
print("\n\n[3] GENERATING QUALITATIVE UMAP PLOTS")

def label_traffic(row):
    if row['type'] == 'extraction_attack': return 'Malicious Extraction Attack'
    elif row['type'] == 'normal_batch': return 'Benign Batch Automation'
    else: return 'Human Conversational Traffic'
df['Traffic_Class'] = df.apply(label_traffic, axis=1)

# --- Plot 1: Binary Decision Boundary (Gateway Action) ---
plt.figure(figsize=(10, 7), facecolor='white')
palette_binary = {'ALLOW': '#2ca02c', 'POISON_PAYLOAD': '#d62728'}
sns.scatterplot(
    x='UMAP_X', y='UMAP_Y', 
    hue='gateway_action', palette=palette_binary, 
    data=df, s=40, alpha=0.7, edgecolor=None
)
plt.title("Gateway Defense (Binary Decision: Allow vs Poison)", fontsize=14, pad=10)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Gateway Action", loc='upper right')
plt.tight_layout()
plot_path_bin = OUTPUT_DIR / "binary_umap.pdf"
plt.savefig(plot_path_bin, format='pdf')
plt.close()

# --- Plot 2: Multiclass Ground Truth (Actual Traffic Types) ---
plt.figure(figsize=(10, 7), facecolor='white')
palette_multi = {
    'Human Conversational Traffic': '#1f77b4', 
    'Benign Batch Automation': '#2ca02c', 
    'Malicious Extraction Attack': '#d62728'
}
sns.scatterplot(
    x='UMAP_X', y='UMAP_Y', 
    hue='Traffic_Class', palette=palette_multi, 
    data=df, s=40, alpha=0.7, edgecolor=None
)
plt.title("Ground Truth Traffic (Multiclass Distribution)", fontsize=14, pad=10)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Traffic Source", loc='upper right')
plt.tight_layout()
plot_path_multi = OUTPUT_DIR / "multiclass_umap.pdf"
plt.savefig(plot_path_multi, format='pdf')
plt.close()

print(f"✓ Binary Plot saved successfully to: {plot_path_bin}")
print(f"✓ Multiclass Plot saved successfully to: {plot_path_multi}")
print("==================================================")
