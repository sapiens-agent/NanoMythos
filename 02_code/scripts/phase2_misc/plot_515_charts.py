import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = '/home/zetyun/nanowhale/reports/515report'
os.makedirs(out_dir, exist_ok=True)

# ===== Plot 1: Length Gradient =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([50, 100, 200, 500, 750], [-0.1155, -0.0549, -0.0013, -0.0073, -0.0008], 'o-', label='Seed 2025', markersize=8)
ax.plot([50, 100, 200, 750], [-0.130, -0.047, -0.002, 0.004], 's--', label='Seed 2027', markersize=8)
ax.plot([50, 100, 750], [-0.075, -0.040, -0.011], '^:', label='Seed 2048', markersize=8)
ax.plot([50, 100, 200, 500, 750], [-0.107, -0.047, -0.002, -0.007, -0.003], 'D-', color='black', linewidth=2.5, label='Mean', markersize=10)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Median text length (chars)', fontsize=13)
ax.set_ylabel('Δ NLL (T=1 − baseline)', fontsize=13)
ax.set_title('Recurrent T=1 advantage vs text length', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks([50, 100, 200, 500, 750])
ax.set_xticklabels(['50', '100', '200', '500', '750'])
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_length_gradient_20260515.png', dpi=150)
print('Plot 1 done')

# ===== Plot 2: T-depth on 50-char =====
nll_vals = [5.970, 5.854, 5.878, 5.902]
delta_t = [0, -0.116, -0.092, -0.068]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(['Baseline', 'T=1', 'T=2', 'T=4'], nll_vals, color=['gray', '#2196F3', '#64B5F6', '#90CAF9'])
ax1.set_ylabel('Strict val NLL', fontsize=12)
ax1.set_title('50-char NLL by recurrent depth', fontsize=13)
for i, v in enumerate(nll_vals):
    ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
ax2.bar(['T=1', 'T=2', 'T=4'], [abs(d) for d in delta_t[1:]], color=['#2196F3', '#64B5F6', '#90CAF9'])
ax2.set_ylabel('|Δ NLL| vs baseline', fontsize=12)
ax2.set_title('Recurrent benefit by depth', fontsize=13)
for i, d in enumerate([abs(d) for d in delta_t[1:]]):
    ax2.text(i, d + 0.002, f'{d:.3f}', ha='center', fontsize=10)
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_tdepth_20260515.png', dpi=150)
print('Plot 2 done')

# ===== Plot 3: Cross-seed stability =====
x = np.arange(4)
s2025 = [-0.116, -0.055, -0.001, -0.001]
s2027 = [-0.130, -0.047, -0.002, 0.004]
s2048 = [-0.075, -0.040, 0, -0.011]
fig, ax = plt.subplots(figsize=(10, 6))
w = 0.25
ax.bar(x - w, s2025, w, label='Seed 2025', color='#2196F3')
ax.bar(x, s2027, w, label='Seed 2027', color='#4CAF50')
ax.bar(x + w, s2048, w, label='Seed 2048', color='#FF9800')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(['50 chars', '100 chars', '200 chars', 'Full'])
ax.set_ylabel('Δ NLL (T=1 − baseline)', fontsize=13)
ax.set_title('Recurrent advantage: cross-seed stability', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
for i in range(4):
    ax.text(i - w, s2025[i] - 0.012, f'{s2025[i]:.3f}', ha='center', fontsize=8)
    ax.text(i, s2027[i] - 0.012, f'{s2027[i]:.3f}', ha='center', fontsize=8)
    if i < 3 or s2048[i] != 0:
        ax.text(i + w, s2048[i] - 0.012, f'{s2048[i]:.3f}', ha='center', fontsize=8)
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_cross_seed_20260515.png', dpi=150)
print('Plot 3 done')

# ===== Plot 4: Packed control =====
fig, ax = plt.subplots(figsize=(8, 5))
settings = ['50-char\nindividual', 'Same content\npacked to 500-char']
deltas = [-0.107, 0.173]
colors = ['#4CAF50', '#F44336']
ax.bar(settings, deltas, color=colors, width=0.5)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.set_ylabel('Δ NLL (T=1 − baseline)', fontsize=13)
ax.set_title('Packed-short control: format drives recurrent effect', fontsize=14)
for i, d in enumerate(deltas):
    ax.text(i, d + (0.02 if d > 0 else -0.04), f'{d:+.3f}', ha='center', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_packed_control_20260515.png', dpi=150)
print('Plot 4 done')

# ===== Plot 5: Mechanism diagnostics =====
metrics = ['hidden_delta\nadjacent', 'hidden_norm\nper_layer', 'moe_load\nimbalance', 'moe_top\nexpert_mass', 'hc_var\nattn_in', 'hc_var\nffn_in']
base_vals = [0.0388, 0.286, 0.295, 0.339, 2.68, 4.13]
t1_vals = [0.0214, 0.254, 0.619, 0.416, 4.83, 9.34]
ratios = [t1_vals[i]/base_vals[i]*100 for i in range(6)]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
ax.bar(x - 0.2, [100]*6, 0.35, label='Baseline (=100%)', color='gray')
ax.bar(x + 0.2, ratios, 0.35, label='T=1 (% of baseline)', color='#2196F3')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylabel('% of baseline value', fontsize=13)
ax.set_title('Mechanism diagnostics: T=1 vs baseline (50-char)', fontsize=14)
ax.legend(fontsize=11)
for i, r in enumerate(ratios):
    color = '#4CAF50' if abs(r - 100) > 30 else 'black'
    y_pos = r + (5 if r > 100 else -8)
    ax.text(i + 0.2, y_pos, f'{r:.0f}%', ha='center', fontsize=9, fontweight='bold', color=color)
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_mechanism_diagnostics_20260515.png', dpi=150)
print('Plot 5 done')

# ===== Plot 6: Epochs vs Δ NLL =====
epochs = [13.5, 7.1, 3.9, 1.8, 1.1]
delta_epoch = [-0.116, -0.055, -0.001, -0.007, -0.001]
labels = ['50', '100', '200', '500', 'Full']

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(epochs, delta_epoch, s=200, color='#2196F3', zorder=5)
for i in range(5):
    ax.annotate(f'{labels[i]} chars', (epochs[i], delta_epoch[i]), 
                textcoords='offset points', xytext=(10, -8), fontsize=10)
z = np.polyfit(epochs, delta_epoch, 1)
p = np.poly1d(z)
x_fit = np.linspace(0, 15, 100)
ax.plot(x_fit, p(x_fit), '--', color='red', alpha=0.5, label=f'Linear fit (R²≈0.94)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Epochs completed (at 3000 steps)', fontsize=13)
ax.set_ylabel('Δ NLL (T=1 − baseline)', fontsize=13)
ax.set_title('Recurrent advantage vs training epochs', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{out_dir}/515_nanowhale_epochs_vs_delta_20260515.png', dpi=150)
print('Plot 6 done')

print('ALL PLOTS DONE')
