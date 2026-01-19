import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = '/mnt/data/'
files = {
    'failrate': base+'exp04_p2_baselines_only_chain_failrate_over_ideal.csv',
    'mae_overall': base+'exp04_p2_baselines_only_chain_mean_over_db(2).csv',
    'mae_only_fail': base+'exp04_p2_baselines_only_chain_mean_over_db_given_over(1).csv'
}

dfs = {k: pd.read_csv(path) for k, path in files.items()}

# Columns are noise levels as strings (e.g., '0', '0.01', ...)
noise_cols = [c for c in dfs['failrate'].columns if c != 'method']
noise_vals = np.array([float(c) for c in noise_cols])

# Ensure consistent method ordering
method_order = dfs['failrate']['method'].tolist()
for k in dfs:
    dfs[k] = dfs[k].set_index('method').loc[method_order].reset_index()

def short_label(m: str) -> str:
    if m == 'A2':
        return 'A2'
    mo = re.match(r'^P2_r(\d+)_(-?\d+)_(-?\d+)_s(\d+)_pareto$', m)
    if mo:
        r, a, b, s = mo.groups()
        return f'r{r}[{a},{b}]_s{s}'
    return m

labels = [short_label(m) for m in method_order]

# IJCAI-ish style, mimicking the provided reference
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.linewidth': 1.5,
})

markers = ['D','o','s','^','v','P','X','*','>','<']

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.0), dpi=300)

panels = [
    ('failrate', '(a) Fail rate', 'Error rate'),
    ('mae_overall', '(b) MAE overall', 'MAE (overall) [dB]'),
    ('mae_only_fail', '(c) MAE only fail', 'MAE (only fail) [dB]'),
]

# Expand y-axis ranges so the curves don't look overly "wavy" due to tight autoscaling.
ylims = {
    'failrate': (0.0, 1.0),
    'mae_overall': (0.0, 10.0),
    'mae_only_fail': (0.0, 15.0),
}

handles = []
for ax, (key, title, ylabel) in zip(axes, panels):
    df = dfs[key]
    for i, (m, lab) in enumerate(zip(method_order, labels)):
        y = df.loc[df['method'] == m, noise_cols].values.squeeze().astype(float)
        # Highlight A2 similar to the highlighted method in the reference figure
        if m == 'A2':
            lw, ms, z = 2.8, 6.5, 6
        else:
            lw, ms, z = 1.8, 5.0, 3
        h = ax.plot(
            noise_vals, y,
            marker=markers[i % len(markers)],
            linewidth=lw,
            markersize=ms,
            zorder=z,
            label=lab,
        )[0]
        if key == 'failrate':
            handles.append(h)

    ax.set_title(title)
    ax.set_xlabel('Noise level (nl)')
    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(noise_vals.min(), noise_vals.max())
    ax.set_xticks(np.arange(0, 0.101, 0.02))
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0, 0.101, 0.02)])

    if key in ylims:
        ax.set_ylim(*ylims[key])

# Legend on the right, like the reference figure
fig.legend(
    handles=handles,
    labels=labels,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    frameon=True,
    borderpad=0.8,
)

fig.subplots_adjust(wspace=0.45, right=0.80, bottom=0.28, top=0.83)

out_pdf = base + 'exp04_p2_only_chain_metrics_ijcai_style_v5.pdf'
out_png = base + 'exp04_p2_only_chain_metrics_ijcai_style_v5.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight')
print('Saved:', out_pdf, out_png)
