import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.5,
    'hatch.linewidth': 2.0
})

labels = ['Qwen-7B', 'Gemma-12B', 'Zysec-7B']
input_tokens = np.array([95.4, 98.9, 101.4])
output_tokens = np.array([30.0, 40.0, 41.1])

x = np.arange(len(labels))
width = 0.45

fig, ax = plt.subplots(figsize=(5.5, 4.5))

p1 = ax.bar(x, input_tokens, width, label='Avg Input Tokens',
            color='#e0e0e0', edgecolor='black', linewidth=1.5, hatch='//')

p2 = ax.bar(x, output_tokens, width, bottom=input_tokens, label='Avg Output Tokens',
            color='#595959', edgecolor='black', linewidth=1.5, hatch='xx')

for i in range(len(labels)):
    total = input_tokens[i] + output_tokens[i]
    ax.text(x[i], total + 3, f'{total:.1f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Total Token Footprint')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0, 160)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.5, linewidth=1.0)
ax.legend(loc='upper left', framealpha=1.0, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig('token_footprint.pdf', format='pdf', dpi=300, bbox_inches='tight')
