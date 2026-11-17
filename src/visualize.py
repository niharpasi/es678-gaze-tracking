import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(
    loso_df: pd.DataFrame,
    calib_curves: dict,
    spatial_df: pd.DataFrame,
    out_dir: str = "./results",
):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
    k_vals = [0, 1, 3, 5, 9]
    markers = ["o", "s", "^"]

    for (model_name, k_dict), marker in zip(calib_curves.items(), markers):
        zero_shot_val = loso_df.loc[loso_df["Model"] == model_name, "LOSO_Mean_mm"].values[0]
        means = [zero_shot_val] + [np.mean(k_dict[k]) for k in [1, 3, 5, 9]]
        stds = [loso_df.loc[loso_df["Model"] == model_name, "LOSO_Std_mm"].values[0]] + [
            np.std(k_dict[k]) / np.sqrt(len(k_dict[k])) for k in [1, 3, 5, 9]
        ]
        ax.errorbar(k_vals, means, yerr=stds, marker=marker, linewidth=2, capsize=3, label=model_name)

    ax.set_xticks(k_vals)
    ax.set_xticklabels(["0\n(Zero-Shot)", "1", "3", "5", "9"])
    ax.set_xlabel("Number of Calibration Fixation Points (k)")
    ax.set_ylabel("Mean Euclidean Gaze Error (mm)")
    ax.set_title("Low-Shot Affine Adaptation Recovery Across Unseen Subjects (LOSO-CV)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_calibration_curve.png"))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    bins_x = np.linspace(0, 1, 7)
    bins_y = np.linspace(0, 1, 5)

    for ax, col, title in zip(
        axes,
        ["err_zero_mm", "err_cal_mm"],
        ["Zero-Shot Kernel Ridge Error (mm)", "5-Point Calibrated Error (mm)"],
    ):
        grid = np.zeros((len(bins_y) - 1, len(bins_x) - 1))
        for iy in range(len(bins_y) - 1):
            for ix in range(len(bins_x) - 1):
                sub = spatial_df[
                    (spatial_df["norm_x"] >= bins_x[ix])
                    & (spatial_df["norm_x"] < bins_x[ix + 1])
                    & (spatial_df["norm_y"] >= bins_y[iy])
                    & (spatial_df["norm_y"] < bins_y[iy + 1])
                ]
                grid[iy, ix] = sub[col].mean() if len(sub) > 0 else np.nan

        im = ax.imshow(grid, origin="upper", aspect="auto", cmap="YlOrRd", vmin=5, vmax=45)
        ax.set_title(title)
        ax.set_xlabel("Normalized Screen X")
        ax.set_ylabel("Normalized Screen Y")
        for iy in range(grid.shape[0]):
            for ix in range(grid.shape[1]):
                if not np.isnan(grid[iy, ix]):
                    ax.text(ix, iy, f"{grid[iy, ix]:.1f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_spatial_heatmap.png"))
    plt.close(fig)
