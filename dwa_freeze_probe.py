#!/usr/bin/env python3
"""
dwa_freeze_probe.py — detect "zero feasible candidate" freeze in DWA after
robot_radius was raised 0.4 -> 0.65 (C4 fix).

Non-invasive: wraps the ALREADY-INSTANTIATED env.dwa.plan bound method to
record its raw (v*, w*) output -- the value chosen BEFORE
BunkerVelocityController's rate limiting. No project file is modified; the
wrap only affects this one object in this one process.

Same fixed start/goal/weights as test_u_tunnel.py so results are directly
comparable.

Usage:
    python3 SAC_Sparse_HER/dwa_freeze_probe.py
    python3 SAC_Sparse_HER/dwa_freeze_probe.py --no-render
"""
import os
import sys
import argparse
import numpy as np
import mujoco

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gymnasium.wrappers import TimeLimit
from gym_bunker_env import BunkerEnv

XML_PATH      = os.path.join(_ROOT, "assets", "worlds", "test", "world_u_tunnel.xml")
ROBOT_START   = [1.1, 9.0, -1.5708]
GOAL_POS      = [4.9, 9.0, 0.0]
WEIGHTS       = np.array([0.55, 0.30, 0.15], dtype=np.float32)  # same as test_u_tunnel.py
MAX_STEPS     = 600
FREEZE_WINDOW = 10   # consecutive raw (0,0) steps -> flagged as a freeze event

# CAVEAT (read before trusting the output blindly): raw (v*,w*)==(0,0) is
# EXACTLY what DWA.plan() returns both when (a) every candidate failed the
# `min_dist <= robot_radius` filter (the fallback default `best_u=[0,0]` was
# never overwritten -- the failure mode we're hunting), AND (b) when (0,0) is
# a genuinely sampled, genuinely-best-scoring candidate (e.g. robot correctly
# judges "don't move, everything nearby is too tight"). This probe cannot
# tell the two apart from the outside. For the practical question "does the
# robot get stuck in the corridor after this change", both cases look and
# matter the same way, so treat "freeze event" as "robot did not move for
# this long", not as proof of the exact code path.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    env = TimeLimit(
        BunkerEnv(xml_path=XML_PATH, render_mode=None if args.no_render else "human",
                  max_goal_sampling_distance=15.0),
        max_episode_steps=MAX_STEPS)
    raw = env.unwrapped
    env.reset()
    raw._goal = np.array(GOAL_POS, dtype=np.float32)
    raw._set_qpos_pose(*ROBOT_START)
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qvel[:] = 0.0
    raw._set_goal_marker_position(raw._goal, is_final_goal=True)

    raw_calls = []
    orig_plan = raw.dwa.plan

    def wrapped_plan(*a, **kw):
        out = orig_plan(*a, **kw)
        raw_calls.append(tuple(out))
        return out

    raw.dwa.plan = wrapped_plan  # instance-level wrap only; DWASolver class untouched

    freeze_run, freeze_events = 0, []
    outcome, step = "TIMEOUT", -1
    for step in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(WEIGHTS)
        v_star, w_star = raw_calls[-1]
        frozen = abs(v_star) < 1e-9 and abs(w_star) < 1e-9

        if frozen:
            freeze_run += 1
        else:
            if freeze_run >= FREEZE_WINDOW:
                freeze_events.append((step - freeze_run, step))
            freeze_run = 0

        if step % 10 == 0 or terminated or truncated:
            dist = float(np.linalg.norm(obs["achieved_goal"][:2] - raw._goal[:2]))
            print(f"step={step:4d}  raw_v*={v_star:+.3f}  raw_w*={w_star:+.3f}  "
                  f"dist={dist:.2f}  frozen_run={freeze_run}")

        if info.get("is_success"):
            outcome = "SUCCESS"; break
        if info.get("collision"):
            outcome = "COLLISION"; break
        if truncated:
            break

    if freeze_run >= FREEZE_WINDOW:
        freeze_events.append((step - freeze_run, step))

    env.close()
    print(f"\noutcome={outcome}  steps={step + 1}")
    print(f"freeze events (raw DWA output == (0,0) for >= {FREEZE_WINDOW} consecutive steps):")
    if not freeze_events:
        print("  none detected")
    else:
        for s0, s1 in freeze_events:
            print(f"  steps {s0}-{s1}  (duration {s1 - s0} steps = {(s1 - s0) * 0.1:.1f}s)")


if __name__ == "__main__":
    main()
