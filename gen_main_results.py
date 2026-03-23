import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bg = '#fdfaf4'

tasks = ['Insert-T', 'Book', 'Towel', 'Scoop', 'Tea', 'Average']
n_tasks = len(tasks)
n_trials = 20
n_real_tasks = 5
max_scores = {'Insert-T': 4, 'Book': 6, 'Towel': 5, 'Scoop': 6, 'Tea': 5}

scores_act       = np.array([0.5125, 0.5750, 0.4921, 0.6292, 0.3750, 0.0])
scores_act_touch = np.array([0.7000, 0.6583, 0.5171, 0.5458, 0.3450, 0.0])
scores_htd       = np.array([0.8125, 0.6917, 0.7129, 0.8417, 0.7175, 0.0])
scores_act[-1]       = np.mean(scores_act[:n_real_tasks])
scores_act_touch[-1] = np.mean(scores_act_touch[:n_real_tasks])
scores_htd[-1]       = np.mean(scores_htd[:n_real_tasks])

std_raw = {
    'Insert-T': [1.2513, 1.4726, 1.6051],
    'Book':     [2.4767, 2.1145, 2.4597],
    'Towel':    [1.6347, 1.4403, 1.6866],
    'Scoop':    [1.7313, 1.9295, 2.0357],
    'Tea':      [1.6489, 0.8807, 0.8867],
}

def compute_sem(task, method_idx):
    if task not in std_raw:
        return 0.0
    return (std_raw[task][method_idx] / max_scores[task]) / np.sqrt(n_trials)

error_act       = np.array([compute_sem(t, 2) for t in tasks[:n_real_tasks]] + [0.0])
error_act_touch = np.array([compute_sem(t, 1) for t in tasks[:n_real_tasks]] + [0.0])
error_htd       = np.array([compute_sem(t, 0) for t in tasks[:n_real_tasks]] + [0.0])
error_act[-1]       = np.mean(error_act[:n_real_tasks])
error_act_touch[-1] = np.mean(error_act_touch[:n_real_tasks])
error_htd[-1]       = np.mean(error_htd[:n_real_tasks])

succ_act       = np.array([0.35, 0.45, 0.20, 0.40, 0.05, 0.0])
succ_act_touch = np.array([0.55, 0.45, 0.20, 0.30, 0.05, 0.0])
succ_htd       = np.array([0.70, 0.60, 0.60, 0.75, 0.50, 0.0])
succ_act[-1]       = np.mean(succ_act[:n_real_tasks])
succ_act_touch[-1] = np.mean(succ_act_touch[:n_real_tasks])
succ_htd[-1]       = np.mean(succ_htd[:n_real_tasks])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
fig.patch.set_facecolor(bg)
ax1.set_facecolor(bg)
ax2.set_facecolor(bg)
bar_width = 0.2
x = np.arange(n_tasks)

color_act = '#a8c8e8'
color_act_touch = '#7fb3de'
color_htd = '#f4b183'

ax1.bar(x - bar_width + 0.03, succ_act, bar_width, color='black', alpha=1.0, zorder=0)
ax1.bar(x + 0.03, succ_act_touch, bar_width, color='black', alpha=1.0, zorder=0)
ax1.bar(x + bar_width + 0.03, succ_htd, bar_width, color='black', alpha=1.0, zorder=0)
ax1.bar(x - bar_width, succ_act, bar_width, label='ACT (Visual + Proprio)', color=color_act, edgecolor='black', zorder=2)
ax1.bar(x, succ_act_touch, bar_width, label='ACT (Visual + Proprio + Touch)', color=color_act_touch, edgecolor='black', zorder=2)
ax1.bar(x + bar_width, succ_htd, bar_width, label='HTD (Ours)', color=color_htd, edgecolor='black', zorder=2)
ax1.set_title('Success Rate', fontsize=14, fontfamily='monospace')
ax1.set_ylabel('Success Rate', fontsize=12, fontfamily='monospace')
ax1.set_xlabel('Task', fontsize=12, fontfamily='monospace')
ax1.set_ylim(0, 1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(['Insert-T', 'Book', 'Towel', 'Scoop', 'Tea', 'Average'], fontsize=11, fontfamily='monospace')
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

ax2.bar(x - bar_width + 0.03, scores_act, bar_width, color='black', alpha=1.0, zorder=0)
ax2.bar(x + 0.03, scores_act_touch, bar_width, color='black', alpha=1.0, zorder=0)
ax2.bar(x + bar_width + 0.03, scores_htd, bar_width, color='black', alpha=1.0, zorder=0)
ax2.bar(x - bar_width, scores_act, bar_width, label='ACT (Visual + Proprio)', color=color_act, edgecolor='black', zorder=2, yerr=error_act, capsize=4)
ax2.bar(x, scores_act_touch, bar_width, label='ACT (Visual + Proprio + Touch)', color=color_act_touch, edgecolor='black', zorder=2, yerr=error_act_touch, capsize=4)
ax2.bar(x + bar_width, scores_htd, bar_width, label='HTD (Ours)', color=color_htd, edgecolor='black', zorder=2, yerr=error_htd, capsize=4)
ax2.set_title('Task Score', fontsize=14, fontfamily='monospace')
ax2.set_ylabel('Task Score', fontsize=12, fontfamily='monospace')
ax2.set_xlabel('Task', fontsize=12, fontfamily='monospace')
ax2.set_ylim(0, 1.0)
ax2.set_xticks(x)
ax2.set_xticklabels(['Insert-T', 'Book', 'Towel', 'Scoop', 'Tea', 'Average'], fontsize=11, fontfamily='monospace')
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

handles, lbls = ax1.get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=13)
plt.subplots_adjust(bottom=0.25)
plt.savefig('figs/htd_main_results.png', dpi=300, bbox_inches='tight', facecolor=bg)
print('Saved figs/htd_main_results.png')
