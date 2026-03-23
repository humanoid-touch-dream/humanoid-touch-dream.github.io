import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

bg = '#fdfaf4'
ymin = 0.3

tasks = ['Insert-T', 'Towel', 'Average']
n_tasks = len(tasks)
n_trials = 20
max_scores = {'Insert-T': 4, 'Towel': 5}

scores_no_touch_td = np.array([0.7500, 0.6643, 0.0])
scores_no_td       = np.array([0.7000, 0.6721, 0.0])
scores_dream_raw   = np.array([0.8250, 0.6593, 0.0])
scores_htd         = np.array([0.8125, 0.7129, 0.0])
scores_no_touch_td[-1] = np.mean(scores_no_touch_td[:2])
scores_no_td[-1]       = np.mean(scores_no_td[:2])
scores_dream_raw[-1]   = np.mean(scores_dream_raw[:2])
scores_htd[-1]         = np.mean(scores_htd[:2])

std_raw = {
    'Insert-T': [1.2513, 0.9787, 1.1965, 1.3765],
    'Towel':    [1.6347, 1.4593, 1.7323, 1.5946],
}

def compute_sem(task, method_idx):
    if task not in std_raw:
        return 0.0
    return (std_raw[task][method_idx] / max_scores[task]) / np.sqrt(n_trials)

error_no_touch_td = np.array([compute_sem('Insert-T', 3), compute_sem('Towel', 3), 0.0])
error_no_td       = np.array([compute_sem('Insert-T', 2), compute_sem('Towel', 2), 0.0])
error_dream_raw   = np.array([compute_sem('Insert-T', 1), compute_sem('Towel', 1), 0.0])
error_htd         = np.array([compute_sem('Insert-T', 0), compute_sem('Towel', 0), 0.0])
error_no_touch_td[-1] = np.mean(error_no_touch_td[:2])
error_no_td[-1]       = np.mean(error_no_td[:2])
error_dream_raw[-1]   = np.mean(error_dream_raw[:2])
error_htd[-1]         = np.mean(error_htd[:2])

succ_no_touch_td = np.array([0.60, 0.45, 0.0])
succ_no_td       = np.array([0.45, 0.55, 0.0])
succ_dream_raw   = np.array([0.65, 0.35, 0.0])
succ_htd         = np.array([0.70, 0.60, 0.0])
succ_no_touch_td[-1] = np.mean(succ_no_touch_td[:2])
succ_no_td[-1]       = np.mean(succ_no_td[:2])
succ_dream_raw[-1]   = np.mean(succ_dream_raw[:2])
succ_htd[-1]         = np.mean(succ_htd[:2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
fig.patch.set_facecolor(bg)
ax1.set_facecolor(bg)
ax2.set_facecolor(bg)
bar_width = 0.18
x = np.arange(n_tasks)
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

color_no_touch_td = '#f7e6a0'
color_no_td       = '#f7d5a0'
color_dream_raw   = '#f7c4a0'
color_htd         = '#f4b183'
colors = [color_no_touch_td, color_no_td, color_dream_raw, color_htd]
all_scores = [scores_no_touch_td, scores_no_td, scores_dream_raw, scores_htd]
all_errors = [error_no_touch_td, error_no_td, error_dream_raw, error_htd]
all_succ   = [succ_no_touch_td, succ_no_td, succ_dream_raw, succ_htd]
labels = ['w/o Touch and TD', 'w/o TD', 'Dream Raw Tactile', 'Dream Latent Tactile']
xtick_labels = ['Insert-T', 'Towel', 'Average']

for off, sc, col, lab in zip(offsets, all_succ, colors, labels):
    ax1.bar(x + off + 0.03, np.maximum(sc - ymin, 0), bar_width, bottom=ymin, color='black', alpha=1.0, zorder=0)
    ax1.bar(x + off, np.maximum(sc - ymin, 0), bar_width, bottom=ymin, label=lab, color=col, edgecolor='black', zorder=2)
ax1.set_title('Success Rate', fontsize=14, fontfamily='monospace')
ax1.set_ylabel('Success Rate', fontsize=12, fontfamily='monospace')
ax1.set_xlabel('Task', fontsize=12, fontfamily='monospace')
ax1.set_ylim(ymin, 0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(xtick_labels, fontsize=11, fontfamily='monospace')
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

d = 0.015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.2)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((-d, +d), (-d - 0.008, +d - 0.008), **kwargs)

for off, sc, err, col, lab in zip(offsets, all_scores, all_errors, colors, labels):
    ax2.bar(x + off + 0.03, np.maximum(sc - ymin, 0), bar_width, bottom=ymin, color='black', alpha=1.0, zorder=0)
    ax2.bar(x + off, np.maximum(sc - ymin, 0), bar_width, bottom=ymin, label=lab, color=col, edgecolor='black', zorder=2, yerr=err, capsize=4)
ax2.set_title('Task Score', fontsize=14, fontfamily='monospace')
ax2.set_ylabel('Task Score', fontsize=12, fontfamily='monospace')
ax2.set_xlabel('Task', fontsize=12, fontfamily='monospace')
ax2.set_ylim(ymin, 0.95)
ax2.set_xticks(x)
ax2.set_xticklabels(xtick_labels, fontsize=11, fontfamily='monospace')
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

kwargs2 = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=1.2)
ax2.plot((-d, +d), (-d, +d), **kwargs2)
ax2.plot((-d, +d), (-d - 0.008, +d - 0.008), **kwargs2)

handles, lbls = ax1.get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=13)
plt.subplots_adjust(bottom=0.25)
plt.savefig('figs/htd_ablation.png', dpi=300, bbox_inches='tight', facecolor=bg)
print('Saved figs/htd_ablation.png')
