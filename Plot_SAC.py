#!/usr/bin/env python3
"""
plot_dwa_sac.py — Vẽ toàn bộ kết quả DWA + SAC từ JSON
---------------------------------------------------------
Chạy sau run_dwa_sac.py. Tự động đọc file JSON và xuất:
  1. Trajectory grid  (tất cả episodes)
  2. Outcome pie chart
  3. SPL + Clearance + Episode Length histograms
  4. SAC weights per episode (α, β, γ theo từng ep)
  5. SAC weights phân phối theo outcome (histogram)
  6. Summary table

Chạy: python3 plot_dwa_sac.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR   not in sys.path: sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from comparison_common import (
    ENV_NAME, RESULTS_DIR, RUN_NAME, load_results
)

# ── Config ───────────────────────────────────────────────────────────
COLOR_SUCCESS   = "#4CAF50"
COLOR_COLLISION = "#F44336"
COLOR_TIMEOUT   = "#FF9800"
COLOR_SAC       = "#1E88E5"
COLOR_ALPHA     = "#7B1FA2"
COLOR_BETA      = "#0288D1"
COLOR_GAMMA     = "#388E3C"


# ════════════════════════════════════════════════════════════════════
# Plot 1: Trajectory Grid
# ════════════════════════════════════════════════════════════════════
def plot_trajectory_grid(episodes, out_path):
    n     = len(episodes)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4 * ncols, 4 * nrows + 0.5))
    axes = axes.flatten()

    color_map = {
        "SUCCESS":   COLOR_SUCCESS,
        "COLLISION": COLOR_COLLISION,
        "TIMEOUT":   COLOR_TIMEOUT,
    }

    for idx, r in enumerate(episodes):
        ax  = axes[idx]
        obs = np.array(r["obstacles"])
        tr  = np.array(r["traj"])

        if len(obs) > 0:
            ax.scatter(obs[:, 0], obs[:, 1], s=1, c="gray", alpha=0.25)

        ax.plot(tr[:, 0], tr[:, 1], "-", color=COLOR_SAC, lw=1.5, alpha=0.85)
        ax.scatter(*r["start"], s=100, c="limegreen",
                   marker="o", edgecolors="k", zorder=5)
        ax.scatter(*r["goal"],  s=130, c="red",
                   marker="*", edgecolors="k", zorder=5)
        ax.add_patch(patches.Circle(tuple(r["goal"]), 0.2,
                                     fill=False, ec="red", ls="--", lw=0.8))

        # Trung bình weights cho ep này
        if r["weights"]:
            w = np.mean(r["weights"], axis=0)
            w_txt = f"α={w[0]:.2f} β={w[1]:.2f} γ={w[2]:.2f}"
        else:
            w_txt = ""

        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        c = color_map.get(r["outcome"], "black")
        ax.set_title(
            f"seed {r['seed']}  |  {r['outcome']}\n"
            f"dist={r['init_dist']:.1f}m  SPL={r['spl']:.2f}  {r['steps']}s\n"
            f"{w_txt}",
            color=c, fontsize=7.5
        )
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"DWA + SAC — Trajectory Grid\n"
        f"World: {ENV_NAME}  |  Model: {RUN_NAME}  |  {n} episodes",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  ✓ Trajectory grid    : {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Plot 2: Outcome Pie Chart
# ════════════════════════════════════════════════════════════════════
def plot_outcome_pie(episodes, out_path):
    n_s = sum(r["outcome"] == "SUCCESS"   for r in episodes)
    n_c = sum(r["outcome"] == "COLLISION" for r in episodes)
    n_t = sum(r["outcome"] == "TIMEOUT"   for r in episodes)
    n   = len(episodes)

    sizes_all  = [n_s, n_c, n_t]
    labels_all = [
        f"SUCCESS\n{n_s}/{n} ({n_s/n*100:.0f}%)",
        f"COLLISION\n{n_c}/{n} ({n_c/n*100:.0f}%)",
        f"TIMEOUT\n{n_t}/{n} ({n_t/n*100:.0f}%)",
    ]
    colors_all = [COLOR_SUCCESS, COLOR_COLLISION, COLOR_TIMEOUT]

    sizes   = [v for v in sizes_all if v > 0]
    labels  = [l for l, v in zip(labels_all, sizes_all) if v > 0]
    colors  = [c for c, v in zip(colors_all, sizes_all) if v > 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")

    ax.set_title(
        f"DWA + SAC — Kết quả {n} episodes\n"
        f"{ENV_NAME}  |  {RUN_NAME}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Outcome pie        : {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Plot 3: SPL + Clearance + Length histograms
# ════════════════════════════════════════════════════════════════════
def plot_metrics_histograms(episodes, out_path):
    spls     = [r["spl"]        for r in episodes]
    clrs     = [r["clearance"]  for r in episodes]
    lens     = [r["steps"]      for r in episodes]
    angs     = [r["ang_smooth"] for r in episodes]
    outcomes = [r["outcome"]    for r in episodes]

    color_per_ep = [
        COLOR_SUCCESS   if o == "SUCCESS"   else
        COLOR_COLLISION if o == "COLLISION" else
        COLOR_TIMEOUT
        for o in outcomes
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    xpos = range(len(spls))

    # SPL bar
    ax = axes[0]
    ax.bar(xpos, spls, color=color_per_ep, edgecolor="k", linewidth=0.4)
    ax.axhline(np.mean(spls), color="navy", lw=1.5, ls="--",
               label=f"Mean = {np.mean(spls):.3f}")
    ax.set_xlabel("Episode index"); ax.set_ylabel("SPL")
    ax.set_title("SPL mỗi episode  (cao = tốt)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Clearance histogram
    ax = axes[1]
    ax.hist(clrs, bins=12, color=COLOR_SAC, edgecolor="k", alpha=0.85)
    ax.axvline(np.mean(clrs), color="navy", lw=1.5, ls="--",
               label=f"Mean = {np.mean(clrs):.2f} m")
    ax.set_xlabel("Mean clearance (m)"); ax.set_ylabel("Số episodes")
    ax.set_title("Phân phối Mean Clearance")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Episode length
    ax = axes[2]
    ax.bar(xpos, lens, color=color_per_ep, edgecolor="k", linewidth=0.4)
    ax.axhline(np.mean(lens), color="navy", lw=1.5, ls="--",
               label=f"Mean = {np.mean(lens):.0f} steps")
    ax.set_xlabel("Episode index"); ax.set_ylabel("Steps")
    ax.set_title("Độ dài mỗi episode")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Angular smoothness
    ax = axes[3]
    ax.hist(angs, bins=12, color=COLOR_SAC, edgecolor="k", alpha=0.85)
    ax.axvline(np.mean(angs), color="navy", lw=1.5, ls="--",
               label=f"Mean = {np.mean(angs):.4f}")
    ax.set_xlabel("Angular smoothness (thấp = mượt hơn)")
    ax.set_ylabel("Số episodes")
    ax.set_title("Phân phối Angular Smoothness")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(fc=COLOR_SUCCESS,   label="SUCCESS"),
        Patch(fc=COLOR_COLLISION, label="COLLISION"),
        Patch(fc=COLOR_TIMEOUT,   label="TIMEOUT"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        f"DWA + SAC — Metrics Breakdown  |  {ENV_NAME}",
        fontsize=13, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Metrics histograms : {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Plot 4: SAC Weights theo từng episode (timeline)
# ════════════════════════════════════════════════════════════════════
def plot_weights_timeline(episodes, out_path):
    """
    Với mỗi episode, lấy mean (α, β, γ).
    Vẽ line chart theo thứ tự episode, màu theo outcome.
    """
    idxs    = []
    alphas  = []
    betas   = []
    gammas  = []
    colors  = []
    markers = []

    for i, r in enumerate(episodes):
        if not r["weights"]:
            continue
        w = np.mean(r["weights"], axis=0)
        idxs.append(i)
        alphas.append(w[0])
        betas.append(w[1])
        gammas.append(w[2])
        if r["outcome"] == "SUCCESS":
            colors.append(COLOR_SUCCESS);   markers.append("o")
        elif r["outcome"] == "COLLISION":
            colors.append(COLOR_COLLISION); markers.append("X")
        else:
            colors.append(COLOR_TIMEOUT);   markers.append("^")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, vals, name, color in zip(
        axes,
        [alphas, betas, gammas],
        ["α (heading)", "β (clearance)", "γ (velocity)"],
        [COLOR_ALPHA, COLOR_BETA, COLOR_GAMMA]
    ):
        ax.plot(idxs, vals, "-", color=color, lw=1.2, alpha=0.5)
        for i, (x, y, c, m) in enumerate(zip(idxs, vals, colors, markers)):
            ax.scatter(x, y, c=c, marker=m, s=60, zorder=5, edgecolors="k",
                       linewidths=0.4)
        ax.axhline(np.mean(vals), color=color, lw=1.5, ls="--",
                   label=f"Mean = {np.mean(vals):.3f}")
        ax.set_ylabel(name, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Episode index", fontsize=11)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SUCCESS,
               markersize=9, label="SUCCESS"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=COLOR_COLLISION,
               markersize=9, label="COLLISION"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLOR_TIMEOUT,
               markersize=9, label="TIMEOUT"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        f"DWA + SAC — Mean Weights mỗi episode  |  {RUN_NAME}  |  {ENV_NAME}",
        fontsize=13, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Weights timeline   : {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Plot 5: SAC Weights Distribution theo outcome
# ════════════════════════════════════════════════════════════════════
def plot_weights_distribution(episodes, out_path):
    by_outcome = {"SUCCESS": [], "COLLISION": [], "TIMEOUT": []}
    for r in episodes:
        if not r["weights"]:
            continue
        w = np.mean(r["weights"], axis=0)
        by_outcome[r["outcome"]].append(w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    w_names = ["α (heading)", "β (clearance)", "γ (velocity)"]
    colors  = {
        "SUCCESS":   COLOR_SUCCESS,
        "COLLISION": COLOR_COLLISION,
        "TIMEOUT":   COLOR_TIMEOUT,
    }

    for i, (ax, wname) in enumerate(zip(axes, w_names)):
        for outcome, ws_list in by_outcome.items():
            if not ws_list:
                continue
            vals = [w[i] for w in ws_list]
            ax.hist(vals, bins=10, alpha=0.6,
                    color=colors[outcome],
                    edgecolor="k", linewidth=0.5,
                    label=f"{outcome} ({len(vals)} ep)")
            ax.axvline(np.mean(vals), color=colors[outcome],
                       lw=2, ls="--")

        ax.set_xlabel(f"Mean {wname} per episode", fontsize=10)
        ax.set_ylabel("Số episodes")
        ax.set_title(f"Phân phối {wname}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"DWA + SAC — Phân phối weights theo outcome  |  {RUN_NAME}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Weights distribution: {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Plot 6: Summary Table
# ════════════════════════════════════════════════════════════════════
def plot_summary_table(episodes, out_path):
    n      = len(episodes)
    sr     = sum(r["outcome"] == "SUCCESS"   for r in episodes) / n * 100
    cr     = sum(r["outcome"] == "COLLISION" for r in episodes) / n * 100
    tr     = sum(r["outcome"] == "TIMEOUT"   for r in episodes) / n * 100
    spl    = np.mean([r["spl"]        for r in episodes])
    clr    = np.mean([r["clearance"]  for r in episodes])
    ang    = np.mean([r["ang_smooth"] for r in episodes])
    lin    = np.mean([r["lin_smooth"] for r in episodes])
    ep_len = np.mean([r["steps"]      for r in episodes])

    # Mean weights qua tất cả episode
    all_w = [np.mean(r["weights"], axis=0)
             for r in episodes if r["weights"]]
    if all_w:
        mean_a = np.mean([w[0] for w in all_w])
        mean_b = np.mean([w[1] for w in all_w])
        mean_g = np.mean([w[2] for w in all_w])
        w_str_a = f"{mean_a:.3f}"
        w_str_b = f"{mean_b:.3f}"
        w_str_g = f"{mean_g:.3f}"
    else:
        w_str_a = w_str_b = w_str_g = "N/A"

    rows = [
        ["Success Rate",      f"{sr:.1f}%",      "↑ cao = tốt"],
        ["Collision Rate",    f"{cr:.1f}%",      "↓ thấp = tốt"],
        ["Timeout Rate",      f"{tr:.1f}%",      "↓ thấp = tốt"],
        ["SPL",               f"{spl:.4f}",      "↑ cao = tốt"],
        ["Mean Clearance",    f"{clr:.3f} m",    "↑ cao = an toàn hơn"],
        ["Angular Smooth",    f"{ang:.5f}",      "↓ thấp = mượt hơn"],
        ["Linear Smooth",     f"{lin:.5f}",      "↓ thấp = mượt hơn"],
        ["Mean Ep. Length",   f"{ep_len:.1f} s", "↓ thấp = nhanh hơn"],
        ["Mean α (heading)",  w_str_a,           "SAC học được"],
        ["Mean β (clearance)",w_str_b,           "SAC học được"],
        ["Mean γ (velocity)", w_str_g,           "SAC học được"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Giá trị", "Ý nghĩa"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.4, 1.9)

    for j in range(3):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        # Highlight rows chứa SAC weights
        if rows[i-1][0].startswith("Mean α") or \
           rows[i-1][0].startswith("Mean β") or \
           rows[i-1][0].startswith("Mean γ"):
            bg = "#E3F2FD"
        else:
            bg = "#FAFAFA" if i % 2 == 0 else "#ECEFF1"
        for j in range(3):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        f"DWA + SAC — Tóm tắt kết quả\n"
        f"World: {ENV_NAME}  |  Model: {RUN_NAME}  |  {n} episodes",
        fontsize=13, fontweight="bold", pad=20
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  ✓ Summary table      : {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    try:
        data = load_results("dwa_sac")
    except FileNotFoundError:
        print("❌ Chưa có kết quả SAC. Hãy chạy run_dwa_sac.py trước!")
        return

    eps = data["episodes"]
    tag = ENV_NAME.replace("/", "_")
    out = os.path.join(RESULTS_DIR, f"dwa_sac_{tag}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n📊 Vẽ kết quả DWA + SAC — {len(eps)} episodes  ({ENV_NAME})")
    print("─" * 60)

    plot_trajectory_grid      (eps, f"{out}_trajectories.png")
    plot_outcome_pie          (eps, f"{out}_outcome_pie.png")
    plot_metrics_histograms   (eps, f"{out}_metrics.png")
    plot_weights_timeline     (eps, f"{out}_weights_timeline.png")
    plot_weights_distribution (eps, f"{out}_weights_dist.png")
    plot_summary_table        (eps, f"{out}_summary_table.png")

    print("─" * 60)
    print(f"✅ Tất cả plots lưu tại: {RESULTS_DIR}/dwa_sac_{tag}_*.png")


if __name__ == "__main__":
    main()