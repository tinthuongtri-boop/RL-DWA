#!/usr/bin/env python3
"""
plot_training_curves.py
─────────────────────────
Đọc TensorBoard logs từ log/run_X/tb/ rồi vẽ training curves PNG để
đánh giá xu hướng huấn luyện đã ổn định chưa.

Output: 6 subplots trong 1 file PNG.

Cách dùng:
    python3 plot_training_curves.py                  # đọc run mới nhất
    python3 plot_training_curves.py --run run_11     # đọc run cụ thể
    python3 plot_training_curves.py --compare 9 10 11 # so sánh nhiều run
"""

import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_event_files(run_dir):
    """Tìm tất cả TensorBoard event files trong run_dir/tb/."""
    pattern = os.path.join(run_dir, "tb", "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def load_tb_scalars(run_dir):
    """
    Đọc tất cả scalar tags từ TensorBoard logs.
    Returns: dict {tag: (steps, values)}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("❌ Cần cài tensorboard: pip install tensorboard")
        sys.exit(1)

    files = find_event_files(run_dir)
    if not files:
        print(f"❌ Không tìm thấy event file trong {run_dir}/tb/")
        return {}

    # EventAccumulator đọc toàn bộ thư mục
    tb_dir = os.path.join(run_dir, "tb")
    # Tìm subfolder run_1/run_2/... bên trong tb/
    subdirs = [d for d in os.listdir(tb_dir)
               if os.path.isdir(os.path.join(tb_dir, d))]
    if subdirs:
        tb_dir = os.path.join(tb_dir, subdirs[0])

    ea = EventAccumulator(tb_dir, size_guidance={'scalars': 0})
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps  = np.array([e.step  for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)
    return data


def smooth(values, window=20):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_single_run(run_dir, run_name, out_path):
    """Vẽ 6 subplot cho 1 run."""
    data = load_tb_scalars(run_dir)
    if not data:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # ── 1. Episode reward ────────────────────────────────────────────
    ax = axes[0]
    if "rollout/ep_rew_mean" in data:
        s, v = data["rollout/ep_rew_mean"]
        ax.plot(s, v, color='steelblue', alpha=0.4, lw=0.8)
        if len(v) > 20:
            sv = smooth(v)
            ss = s[len(s)-len(sv):]
            ax.plot(ss, sv, color='steelblue', lw=2.0, label='Smoothed (w=20)')
        ax.set_xlabel('Timesteps'); ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Episode Reward (xu hướng tổng)')
        ax.grid(alpha=0.3); ax.legend()

    # ── 2. Outcome rates ─────────────────────────────────────────────
    ax = axes[1]
    for tag, color, label in [
        ("custom/success_rate",   "green",  "Success"),
        ("custom/collision_rate", "red",    "Collision"),
        ("custom/stuck_rate",     "orange", "Stuck"),
        ("custom/timeout_rate",   "gray",   "Timeout"),
    ]:
        if tag in data:
            s, v = data[tag]
            ax.plot(s, v, color=color, lw=1.5, label=label, alpha=0.85)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Rate')
    ax.set_title('Outcome Rates (tăng SR / giảm CR là tốt)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ── 3. SAC weights mean ──────────────────────────────────────────
    ax = axes[2]
    for tag, color, label in [
        ("custom/weight_alpha_mean", "purple",  "α (heading)"),
        ("custom/weight_beta_mean",  "blue",    "β (clearance)"),
        ("custom/weight_gamma_mean", "green",   "γ (velocity)"),
    ]:
        if tag in data:
            s, v = data[tag]
            ax.plot(s, v, color=color, lw=1.5, label=label)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Mean weight value')
    ax.set_title('SAC Weight Means (xu hướng các trọng số)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ── 4. SAC weights std (mức độ adaptive) ─────────────────────────
    ax = axes[3]
    for tag, color, label in [
        ("custom/weight_alpha_std", "purple", "α std"),
        ("custom/weight_beta_std",  "blue",   "β std"),
        ("custom/weight_gamma_std", "green",  "γ std"),
    ]:
        if tag in data:
            s, v = data[tag]
            ax.plot(s, v, color=color, lw=1.5, label=label)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Std deviation')
    ax.set_title('Weight Variance (cao = adaptive, thấp = degenerate)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ── 5. Behavior metrics: v, |w|, clearance ───────────────────────
    ax = axes[4]
    if "custom/mean_v" in data:
        s, v = data["custom/mean_v"]
        ax.plot(s, v, color='blue',   lw=1.5, label='Mean v (m/s)')
    if "custom/mean_abs_w" in data:
        s, v = data["custom/mean_abs_w"]
        ax.plot(s, v, color='purple', lw=1.5, label='Mean |ω| (rad/s)')
    if "custom/mean_min_clearance" in data:
        s, v = data["custom/mean_min_clearance"]
        ax.plot(s, v / 5.0, color='orange', lw=1.5,
                 label='Mean clearance / 5 (m)')
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Magnitude')
    ax.set_title('Behavior metrics (vận tốc / xoay / clearance)')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ── 6. SAC training losses (actor / critic / entropy) ────────────
    ax = axes[5]
    for tag, color, label in [
        ("train/actor_loss",   "blue",   "Actor loss"),
        ("train/critic_loss",  "red",    "Critic loss"),
        ("train/ent_coef",     "green",  "Entropy coef"),
    ]:
        if tag in data:
            s, v = data[tag]
            ax.plot(s, v, color=color, lw=1.2, label=label, alpha=0.85)
    ax.set_xlabel('Timesteps'); ax.set_ylabel('Value')
    ax.set_title('SAC Training Losses')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_yscale('symlog')

    fig.suptitle(f"Training Curves — {run_name}", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close()


def plot_compare(run_dirs, run_names, out_path):
    """So sánh nhiều run trên cùng 1 plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes  = axes.flatten()
    cmap  = plt.cm.tab10

    panels = [
        ("rollout/ep_rew_mean",      "Episode Reward",         "Mean reward"),
        ("custom/success_rate",      "Success Rate",            "Success rate"),
        ("custom/collision_rate",    "Collision Rate",          "Collision rate"),
        ("custom/weight_alpha_mean", "α (heading) Mean",        "α value"),
        ("custom/weight_beta_mean",  "β (clearance) Mean",      "β value"),
        ("custom/weight_gamma_mean", "γ (velocity) Mean",       "γ value"),
    ]

    for ax, (tag, title, ylabel) in zip(axes, panels):
        for i, (rd, rn) in enumerate(zip(run_dirs, run_names)):
            data = load_tb_scalars(rd)
            if tag in data:
                s, v = data[tag]
                ax.plot(s, v, color=cmap(i), lw=1.5, label=rn, alpha=0.85)
        ax.set_xlabel('Timesteps'); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle("Training Curves — So sánh nhiều run",
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"✓ Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",     help="Tên run cụ thể (vd: run_11)")
    parser.add_argument("--compare", nargs='+', type=str,
                        help="So sánh nhiều run (vd: --compare 9 10 11)")
    parser.add_argument("--log-dir", default=os.path.join(_SCRIPT_DIR, "log"),
                        help="Thư mục chứa các run")
    args = parser.parse_args()

    log_dir = args.log_dir

    if args.compare:
        # So sánh nhiều run
        run_dirs  = []
        run_names = []
        for r in args.compare:
            name = f"run_{r}" if not r.startswith("run_") else r
            path = os.path.join(log_dir, name)
            if os.path.isdir(path):
                run_dirs.append(path); run_names.append(name)
            else:
                print(f"⚠ Bỏ qua {name}: không tìm thấy {path}")

        if not run_dirs:
            print("❌ Không có run nào hợp lệ.")
            return

        out_path = os.path.join(log_dir, f"compare_{'_'.join(run_names)}.png")
        plot_compare(run_dirs, run_names, out_path)
        return

    # Chọn run
    if args.run:
        run_name = args.run if args.run.startswith("run_") else f"run_{args.run}"
        run_dir  = os.path.join(log_dir, run_name)
    else:
        # Lấy run mới nhất
        runs = sorted([d for d in os.listdir(log_dir)
                       if d.startswith("run_") and
                       os.path.isdir(os.path.join(log_dir, d))],
                      key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        if not runs:
            print(f"❌ Không tìm thấy run nào trong {log_dir}")
            return
        run_name = runs[-1]
        run_dir  = os.path.join(log_dir, run_name)
        print(f"📂 Sử dụng run mới nhất: {run_name}")

    if not os.path.isdir(run_dir):
        print(f"❌ Không tìm thấy: {run_dir}")
        return

    out_path = os.path.join(run_dir, "training_curves.png")
    plot_single_run(run_dir, run_name, out_path)


if __name__ == "__main__":
    main()
