import matplotlib.pyplot as plt
import numpy as np

# --- 兼容老版 matplotlib ---
try:
    from matplotlib.colors import DivergingNorm as _DivNorm
except ImportError:
    from matplotlib.colors import TwoSlopeNorm as _DivNorm   # 老版本 fallback

def five_panel(rho_c, rho_mid, rho_p, *,
               labels=("Classic", "Mid", "Δ", "Quantum+", "Δ+"),
               title="", figsize=(9, 2.3)):
    diff   = rho_mid - rho_c
    diff_p = rho_p   - rho_c

    vmin, vmax = rho_c.min(), rho_c.max()
    dmax = max(abs(diff).max(), abs(diff_p).max())
    if dmax == 0:            # 全零残差 → 给个极小幅度
        dmax = 1e-12
    norm = _DivNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)


    fig, axes = plt.subplots(1, 5, figsize=figsize,
                             constrained_layout=True,
                             sharex=True, sharey=True)

    # 连续三幅
    for ax, dat, lbl in zip([axes[0], axes[1], axes[3]],
                            [rho_c,   rho_mid,  rho_p],
                            [labels[0], labels[1], labels[3]]):
        im = ax.pcolormesh(dat.real, cmap="viridis", shading="auto",
                           vmin=vmin, vmax=vmax)
        ax.set_title(lbl, fontsize=8); ax.axis("off")

    # 差分两幅
    for ax, dat, lbl in zip([axes[2], axes[4]], [diff, diff_p],
                            [labels[2], labels[4]]):
        im = ax.pcolormesh(dat.real, cmap="coolwarm", norm=norm, shading="auto")
        ax.set_title(lbl, fontsize=8); ax.axis("off")

    fig.colorbar(im, ax=axes, location="right", shrink=0.8, pad=0.02)
    fig.suptitle(title, fontsize=10)
    return fig
