#!/usr/bin/env python3
"""
plot_comparison.py — So sánh trực quan 2 method từ JSON results
-----------------------------------------------------------------
Đọc kết quả từ run_dwa_only.py và run_dwa_sac.py, vẽ:
  1. Trajectory grid  — N cặp ep cạnh nhau
  2. Metrics bar chart — SR/CR/SPL/Clearance
  3. SAC weights timeline — SAC đã học chọn weights thế nào
  4. Text summary
"""

import os
import numpy as np

from comparison_common import (
    ENV_NAME, RESULTS_DIR, RUN_NAME, BASELINE_WEIGHTS,
    load_results,
)


def plot_trajectory_grid(dwa_episodes, sac_episodes,
                          out_path, n_plot=6):
    """Grid 2 hàng × n_plot cột: trên = DWA, dưới = DWA-SAC."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n_plot = min(n_plot, len(dwa_episodes), len(sac_episodes))
    fig, axes = plt.subplots(2, n_plot, figsize=(4*n_plot, 8.5))
    if n_plot == 1:
        axes = axes.reshape(2, 1)

    color_map = {'SUCCESS':'green', 'COLLISION':'red', 'TIMEOUT':'orange'}

    for col in range(n_plot):
        for row, (eps, label) in enumerate(
                [(dwa_episodes, 'DWA'),
                 (sac_episodes, 'DWA + SAC')]):

            r  = eps[col]
            ax = axes[row, col]

            # LiDAR hits
            obs = np.array(r["obstacles"])
            if len(obs) > 0:
                ax.scatter(obs[:,0], obs[:,1], s=1, c='gray', alpha=0.3)

            # Trajectory
            tr = np.array(r["traj"])
            ax.plot(tr[:,0], tr[:,1], '-', color='blue', lw=1.8, alpha=0.8)

            # Start (green circle) & Goal (red star)
            ax.scatter(*r["start"], s=120, c='limegreen',
                       marker='o', edgecolors='k', zorder=5)
            ax.scatter(*r["goal"],  s=150, c='red',
                       marker='*', edgecolors='k', zorder=5)
            ax.add_patch(patches.Circle(tuple(r["goal"]), 0.2,
                                         fill=False, ec='red', ls='--'))

            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.set_title(
                f"{label}  (seed {r['seed']})\n"
                f"{r['outcome']} • {r['steps']}s • SPL={r['spl']:.2f}",
                color=color_map.get(r['outcome'], 'k'), fontsize=10)

    plt.suptitle(f"Trajectory Comparison — {ENV_NAME}",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"📊 Trajectory grid saved: {out_path}")
    plt.close()


def plot_metrics_bar(dwa_episodes, sac_episodes, out_path):
    """Bar chart so sánh các metric quan trọng."""
    import matplotlib.pyplot as plt

    def aggregate(eps):
        n   = len(eps)
        sr  = sum(r['outcome']=='SUCCESS'   for r in eps) / n
        cr  = sum(r['outcome']=='COLLISION' for r in eps) / n
        tr  = sum(r['outcome']=='TIMEOUT'   for r in eps) / n
        spl = np.mean([r['spl']        for r in eps])
        clr = np.mean([r['clearance']  for r in eps])
        ang = np.mean([r['ang_smooth'] for r in eps])
        return [sr, cr, tr, spl, clr, ang]

    dwa_vals = aggregate(dwa_episodes)
    sac_vals = aggregate(sac_episodes)

    labels = ['Success\nRate', 'Collision\nRate', 'Timeout\nRate',
              'SPL\n(High = Good)', 'Clearance\n(m)', 'Ang.Smooth\n(Low = Good)']

    # Normalize clearance và ang.smooth để hiển thị cùng scale [0,1]
    # (chỉ để visual, số hiển thị là giá trị thật)
    max_clr = max(dwa_vals[4], sac_vals[4], 1.0)
    max_ang = max(dwa_vals[5], sac_vals[5], 0.1)

    dwa_scaled = [dwa_vals[0], dwa_vals[1], dwa_vals[2], dwa_vals[3],
                   dwa_vals[4]/max_clr, dwa_vals[5]/max_ang]
    sac_scaled = [sac_vals[0], sac_vals[1], sac_vals[2], sac_vals[3],
                   sac_vals[4]/max_clr, sac_vals[5]/max_ang]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bar_dwa = ax.bar(x - w/2, dwa_scaled, w,
                     label=f'DWA (α={BASELINE_WEIGHTS[0]} β={BASELINE_WEIGHTS[1]} γ={BASELINE_WEIGHTS[2]})',
                     color='coral')
    bar_sac = ax.bar(x + w/2, sac_scaled, w,
                     label=f'DWA + SAC ({RUN_NAME})', color='royalblue')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Normalized Value')
    ax.set_title(f"Metrics Comparision — {ENV_NAME}  ({len(dwa_episodes)} episodes, with seed)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.15)

    # Ghi giá trị thật lên đầu bar
    for bars, real_vals in [(bar_dwa, dwa_vals), (bar_sac, sac_vals)]:
        for bar, v in zip(bars, real_vals):
            ax.annotate(f'{v:.3f}',
                        xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"📊 Metrics bar chart saved: {out_path}")
    plt.close()


def plot_sac_weights(sac_episodes, out_path):
    """Phân phối weights SAC chọn qua các episode."""
    import matplotlib.pyplot as plt

    # Gom weights theo outcome
    all_weights = {'SUCCESS': [], 'COLLISION': [], 'TIMEOUT': []}
    for r in sac_episodes:
        if not r['weights']:
            continue
        mean_w = np.mean(r['weights'], axis=0)
        all_weights[r['outcome']].append(mean_w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    weight_names = ['α (heading)', 'β (clearance)', 'γ (velocity)']
    colors = {'SUCCESS':'green', 'COLLISION':'red', 'TIMEOUT':'orange'}

    for i, name in enumerate(weight_names):
        ax = axes[i]
        for outcome, ws_list in all_weights.items():
            if not ws_list:
                continue
            vals = [w[i] for w in ws_list]
            ax.hist(vals, bins=12, alpha=0.5,
                    color=colors[outcome],
                    label=f'{outcome} ({len(vals)})')
        ax.set_xlabel(f'Mean {name} per episode')
        ax.set_ylabel('Number of episodes')
        ax.set_title(f'Distribution {name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"SAC Policy Weights Distribution — {RUN_NAME}",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"📊 SAC weights distribution saved: {out_path}")
    plt.close()


def write_text_summary(dwa_ep, sac_ep, out_path):
    def stats(eps):
        n   = len(eps)
        return {
            'n':    n,
            'sr':   sum(r['outcome']=='SUCCESS'   for r in eps) / n * 100,
            'cr':   sum(r['outcome']=='COLLISION' for r in eps) / n * 100,
            'tr':   sum(r['outcome']=='TIMEOUT'   for r in eps) / n * 100,
            'spl':  np.mean([r['spl']        for r in eps]),
            'clr':  np.mean([r['clearance']  for r in eps]),
            'ang':  np.mean([r['ang_smooth'] for r in eps]),
            'lin':  np.mean([r['lin_smooth'] for r in eps]),
            'len':  np.mean([r['steps']      for r in eps]),
        }

    d = stats(dwa_ep)
    s = stats(sac_ep)

    lines = [
        "=" * 72,
        f"  SO SÁNH DWA THUẦN vs DWA + SAC  —  {ENV_NAME}",
        f"  {d['n']} episodes, cùng seed → cùng start/goal",
        "=" * 72,
        f"{'Metric':<20} {'DWA thuần':>15} {'DWA + SAC':>15} {'Δ':>15}",
        "-" * 72,
        f"{'Success Rate (%)':<20} {d['sr']:>14.1f}  {s['sr']:>14.1f}  {s['sr']-d['sr']:>+14.1f}",
        f"{'Collision Rate (%)':<20} {d['cr']:>14.1f}  {s['cr']:>14.1f}  {s['cr']-d['cr']:>+14.1f}",
        f"{'Timeout Rate (%)':<20} {d['tr']:>14.1f}  {s['tr']:>14.1f}  {s['tr']-d['tr']:>+14.1f}",
        f"{'SPL':<20} {d['spl']:>14.3f}  {s['spl']:>14.3f}  {s['spl']-d['spl']:>+14.3f}",
        f"{'Clearance (m)':<20} {d['clr']:>14.3f}  {s['clr']:>14.3f}  {s['clr']-d['clr']:>+14.3f}",
        f"{'Ang.Smooth':<20} {d['ang']:>14.4f}  {s['ang']:>14.4f}  {s['ang']-d['ang']:>+14.4f}",
        f"{'Lin.Smooth':<20} {d['lin']:>14.4f}  {s['lin']:>14.4f}  {s['lin']-d['lin']:>+14.4f}",
        f"{'Mean Ep. Length':<20} {d['len']:>14.1f}  {s['len']:>14.1f}  {s['len']-d['len']:>+14.1f}",
        "=" * 72,
        "",
        "Ghi chú:",
        "  • Δ = (SAC - DWA). Âm = SAC tốt hơn (cho CR, Ang, Lin, Len).",
        "  • SR, SPL, Clearance: dương = SAC tốt hơn.",
    ]

    txt = "\n".join(lines)
    print(txt)
    with open(out_path, "w") as f:
        f.write(txt)
    print(f"\n📝 Summary saved: {out_path}")


def main():
    # Load cả 2 results
    try:
        dwa_data = load_results("dwa_only")
        sac_data = load_results("dwa_sac")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Hãy chạy run_dwa_only.py và run_dwa_sac.py trước!")
        return

    dwa_ep = dwa_data["episodes"]
    sac_ep = sac_data["episodes"]

    tag = ENV_NAME.replace("/", "_")

    write_text_summary(dwa_ep, sac_ep,
        os.path.join(RESULTS_DIR, f"summary_{tag}.txt"))

    plot_trajectory_grid(dwa_ep, sac_ep,
        os.path.join(RESULTS_DIR, f"plot_trajectories_{tag}.png"))

    plot_metrics_bar(dwa_ep, sac_ep,
        os.path.join(RESULTS_DIR, f"plot_metrics_{tag}.png"))

    plot_sac_weights(sac_ep,
        os.path.join(RESULTS_DIR, f"plot_sac_weights_{tag}.png"))

    print(f"\n✅ Tất cả kết quả so sánh lưu tại:  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()