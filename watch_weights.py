#!/usr/bin/env python3
"""
watch_weights.py — Run ONE episode and watch the DWA weights in real time.

Put this file next to test_u_tunnel.py (SAC_Sparse_HER/) and run from the project root:

    python3 SAC_Sparse_HER/watch_weights.py --world val/World_Medium --seed 1024
    python3 SAC_Sparse_HER/watch_weights.py --ckpt SAC_Sparse_HER/log/run_13/best_model_backup_gd3.zip
    python3 SAC_Sparse_HER/watch_weights.py --fixed 0.7 0.2 0.1        # DWA-only baseline
    python3 SAC_Sparse_HER/watch_weights.py --live --every 2            # live matplotlib window
    python3 SAC_Sparse_HER/watch_weights.py --no-render --stochastic    # headless, sampled actions

What is printed per row (every --every steps):
    raw a,b,g      : SAC output in [0,1]^3 (what the env receives)
    w_head/clear/vel: the SAME numbers after L1-normalisation. This is what DWA actually
                     uses (DWA.plan divides by the sum), so THIS is the meaningful quantity.
    mode           : ROT = v*=0 (rotate-in-place gate of DWA), DRV = forward, REV = reverse
    v*, w*         : DWA command that was sent to the low-level controller
    v, w[5], w[1]  : measured forward speed, and yaw rate read from qvel[5] and from the index
                     the env currently uses (angular_body[1]). If w[1] stays ~0 while w[5]
                     moves, the omega-axis bug (review item C2) is confirmed.
"""

import os
import sys
import csv
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gymnasium.wrappers import TimeLimit
from gym_bunker_env import BunkerEnv


# ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="val/World_Medium",
                   help="path under assets/worlds without .xml")
    p.add_argument("--run", default="run_13")
    p.add_argument("--ckpt", default=None,
                   help="explicit .zip path (overrides --run), e.g. a best_model_backup_*.zip")
    p.add_argument("--fixed", nargs=3, type=float, default=None, metavar=("A", "B", "G"),
                   help="skip SAC and use constant DWA weights")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--start", nargs=3, type=float, default=None, metavar=("X", "Y", "TH"))
    p.add_argument("--goal", nargs=3, type=float, default=None, metavar=("X", "Y", "TH"))
    p.add_argument("--max-goal-dist", type=float, default=12.0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--every", type=int, default=5, help="print every N steps")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--live", action="store_true", help="live matplotlib window")
    p.add_argument("--no-plot", action="store_true", help="do not save the final PNG")
    return p.parse_args()


def load_policy(env, args):
    """Load SAC. HER needs an env at load time, so wrap the SAME env object once."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    ckpt = args.ckpt or os.path.join(_SCRIPT_DIR, "log", args.run,
                                     "best_model", "best_model.zip")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    used = [False]

    def _factory():
        if used[0]:
            raise RuntimeError("factory may only be called once")
        used[0] = True
        return env

    model = SAC.load(ckpt, env=DummyVecEnv([_factory]), device="auto")
    print(f"[watch] SAC loaded: {ckpt}")

    def policy(obs):
        obs_b = {k: np.asarray(v)[np.newaxis, :] for k, v in obs.items()}
        act, _ = model.predict(obs_b, deterministic=not args.stochastic)
        return act[0].astype(np.float32)

    return policy


# ──────────────────────────────────────────────────────────────────────
class Plot:
    """3 stacked axes: normalised weights / velocities / distances."""

    def __init__(self, live):
        import matplotlib.pyplot as plt
        self.plt = plt
        if live:
            plt.ion()
        self.fig, self.ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        a0, a1, a2 = self.ax
        self.lines = {}
        for k, c in (("w_head", "royalblue"), ("w_clear", "darkorange"), ("w_vel", "green")):
            (self.lines[k],) = a0.plot([], [], color=c, label=k)
        a0.set_ylim(0, 1.05)
        a0.set_ylabel("normalised weight")
        a0.legend(loc="upper right")
        for k, c, ls in (("v_cmd", "navy", "-"), ("v", "navy", ":"),
                         ("w_cmd", "purple", "-"), ("w5", "purple", ":")):
            (self.lines[k],) = a1.plot([], [], color=c, ls=ls, label=k)
        a1.axhline(0, color="k", lw=0.5)
        a1.set_ylabel("m/s | rad/s")
        a1.legend(loc="upper right", ncol=4)
        for k, c in (("dist_goal", "crimson"), ("min_r", "gray")):
            (self.lines[k],) = a2.plot([], [], color=c, label=k)
        a2.set_ylabel("m")
        a2.set_xlabel("t (s)")
        a2.legend(loc="upper right")
        for a in self.ax:
            a.grid(alpha=0.3)

    def update(self, log, pause=0.001):
        t = log["t"]
        for k, line in self.lines.items():
            line.set_data(t, log[k])
        for a in self.ax:
            a.relim()
            a.autoscale_view(scalex=True, scaley=(a is not self.ax[0]))
        self.fig.canvas.draw_idle()
        if pause:
            self.plt.pause(pause)

    def save(self, path, title):
        self.fig.suptitle(title, fontweight="bold")
        self.fig.savefig(path, dpi=130, bbox_inches="tight")


# ──────────────────────────────────────────────────────────────────────
HEADER = (f"{'step':>5} {'t':>6} {'x':>6} {'y':>6} {'dist':>6} {'min_r':>6} | "
          f"{'a':>5} {'b':>5} {'g':>5} | {'w_head':>6} {'w_clr':>6} {'w_vel':>6} | "
          f"{'mode':>4} {'v*':>6} {'w*':>6} | {'v':>6} {'w[5]':>6} {'w[1]':>6}")


def main():
    args = parse_args()
    xml = os.path.join(_PROJECT_ROOT, "assets", "worlds", f"{args.world}.xml")
    if not os.path.exists(xml):
        sys.exit(f"world not found: {xml}")

    env = TimeLimit(
        BunkerEnv(xml_path=xml,
                  render_mode=None if args.no_render else "human",
                  max_goal_sampling_distance=args.max_goal_dist),
        max_episode_steps=args.max_steps)
    raw = env.unwrapped
    dt = raw.model.opt.timestep * raw.frame_skip

    if args.fixed is not None:
        fixed = np.asarray(args.fixed, dtype=np.float32)
        policy = lambda obs: fixed
        tag = "fixed_%g_%g_%g" % tuple(args.fixed)
    else:
        policy = load_policy(env, args)
        tag = args.ckpt and os.path.splitext(os.path.basename(args.ckpt))[0] or args.run

    obs, _ = env.reset(seed=args.seed)
    if args.start is not None and args.goal is not None:
        m = raw.set_manual_pose(args.start, args.goal)   # raises ValueError if in collision
        obs = m
    print(f"[watch] world={args.world} seed={args.seed} dt={dt:.2f}s "
          f"start={obs['achieved_goal'][:2].round(2)} goal={obs['desired_goal'][:2].round(2)}")
    print(HEADER)
    print("-" * len(HEADER))

    keys = ["t", "x", "y", "dist_goal", "min_r", "a", "b", "g",
            "w_head", "w_clear", "w_vel", "v_cmd", "w_cmd", "v", "w5", "w1"]
    log = {k: [] for k in keys}
    plot = Plot(live=args.live) if (args.live or not args.no_plot) else None

    outcome = "TIMEOUT"
    for step in range(args.max_steps):
        action = np.asarray(policy(obs), dtype=np.float32)
        a, b, g = (float(action[0]), float(action[1]), float(action[2]))

        # Same normalisation as DWASolver.plan(): weights are divided by their sum.
        s = a + b + g
        wn = (a / s, b / s, g / s) if s > 1e-6 else (1 / 3, 1 / 3, 1 / 3)

        obs, _, terminated, truncated, info = env.step(action)

        vc = raw.velocity_controller                    # holds the last DWA command
        v_cmd, w_cmd = float(vc.v_cmd), float(vc.w_cmd)
        lin_b, ang_b = raw.get_robot_velocities()
        v_meas, w1 = float(lin_b[0]), float(ang_b[1])   # w1 = what the env feeds to SAC
        w5 = float(raw.data.qvel[5])                    # yaw rate (local z, robot is upright)
        x, y = float(obs["achieved_goal"][0]), float(obs["achieved_goal"][1])
        dist = float(np.linalg.norm(obs["achieved_goal"][:2] - obs["desired_goal"][:2]))
        min_r = float(info.get("min_lidar", np.nan))
        mode = "ROT" if abs(v_cmd) < 1e-3 else ("REV" if v_cmd < 0 else "DRV")

        row = dict(t=step * dt, x=x, y=y, dist_goal=dist, min_r=min_r, a=a, b=b, g=g,
                   w_head=wn[0], w_clear=wn[1], w_vel=wn[2], v_cmd=v_cmd, w_cmd=w_cmd,
                   v=v_meas, w5=w5, w1=w1)
        for k in keys:
            log[k].append(row[k])

        end = terminated or truncated
        if step % args.every == 0 or end:
            if step and step % (args.every * 20) == 0:
                print(HEADER)
            print(f"{step:>5} {row['t']:>6.1f} {x:>6.2f} {y:>6.2f} {dist:>6.2f} {min_r:>6.2f} | "
                  f"{a:>5.2f} {b:>5.2f} {g:>5.2f} | {wn[0]:>6.2f} {wn[1]:>6.2f} {wn[2]:>6.2f} | "
                  f"{mode:>4} {v_cmd:>+6.2f} {w_cmd:>+6.2f} | {v_meas:>+6.2f} {w5:>+6.2f} {w1:>+6.2f}",
                  flush=True)
            if args.live and plot is not None:
                plot.update(log)

        if info.get("is_success"):
            outcome = "SUCCESS"
        elif info.get("collision"):
            outcome = "COLLISION"
        elif info.get("stuck"):
            outcome = "STUCK"
        if end:
            break

    env.close()

    # ── summary ───────────────────────────────────────────────────────
    W = np.array([log["w_head"], log["w_clear"], log["w_vel"]])
    print("-" * len(HEADER))
    print(f"[watch] outcome={outcome}  steps={len(log['t'])}  path-time={log['t'][-1] + dt:.1f}s")
    print("[watch] normalised weights  mean=" + np.array2string(W.mean(1), precision=3) +
          "  std=" + np.array2string(W.std(1), precision=3) +
          "  min=" + np.array2string(W.min(1), precision=3) +
          "  max=" + np.array2string(W.max(1), precision=3))
    print(f"[watch] |w[5]| mean={np.mean(np.abs(log['w5'])):.3f}   "
          f"|w[1]| mean={np.mean(np.abs(log['w1'])):.3f}   (w[1]~0 while w[5]>0 => C2 confirmed)")

    out_dir = os.path.join(_SCRIPT_DIR, "comparison_results")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"watch_{tag}_{args.world.replace('/', '_')}_s{args.seed}")
    with open(stem + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        wr.writerows(zip(*[log[k] for k in keys]))
    print(f"[watch] CSV : {stem}.csv")
    if plot is not None and not args.no_plot:
        plot.update(log, pause=0)
        plot.save(stem + ".png", f"{tag} | {args.world} | seed {args.seed} | {outcome}")
        print(f"[watch] PNG : {stem}.png")


if __name__ == "__main__":
    main()
