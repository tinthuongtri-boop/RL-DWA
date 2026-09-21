#!/usr/bin/env python3
"""
run_dwa_only.py — Chạy DWA thuần với weights CỐ ĐỊNH
------------------------------------------------------
Dùng SEED cố định → start/goal y hệt như khi chạy run_dwa_sac.py.
Lưu kết quả vào JSON để script plot_comparison.py dùng sau.
"""

import numpy as np
from gymnasium.wrappers import TimeLimit

from comparison_common import (
    BASELINE_WEIGHTS, MAX_GOAL_DIST, MAX_EP_STEPS,
    get_xml_path, get_seeds,
    run_single_episode, save_results, print_summary,
)
from gym_bunker_env import BunkerEnv
from dynamic_obstacles_controller import DynamicObstacleController


def main():
    alpha, beta, gamma = BASELINE_WEIGHTS
    action_fixed = np.array([alpha, beta, gamma], dtype=np.float32)

    # Policy: luôn trả về weights cố định
    def dwa_policy(obs):
        return action_fixed

    # ── Tạo env ───────────────────────────────────────────────────
    env = TimeLimit(
        BunkerEnv(xml_path=get_xml_path(), render_mode= "rgb_array",
                  max_goal_sampling_distance=MAX_GOAL_DIST),
        max_episode_steps=MAX_EP_STEPS,
    )

    seeds   = get_seeds()
    results = []


    print(f"╔{'═'*58}╗")
    print(f"║  RUN 1: DWA THUẦN                                        ║")
    print(f"║  Weights cố định: α={alpha}  β={beta}  γ={gamma}               ║")
    print(f"║  {len(seeds)} episodes                                              ║")
    print(f"╚{'═'*58}╝\n")

    for i, seed in enumerate(seeds):
        r = run_single_episode(env, dwa_policy, seed=seed)
        results.append(r)
        print(f"  EP {i+1:2d}/{len(seeds)}  seed={seed}  "
              f"{r['outcome']:10s}  {r['steps']:3d} steps  "
              f"dist={r['init_dist']:4.1f}m  SPL={r['spl']:.2f}")

    env.close()

    # Lưu kết quả
    save_results(results, method_name="dwa_only",
                 extra_meta={"weights": list(BASELINE_WEIGHTS)})

    print_summary("DWA THUẦN", results)


if __name__ == "__main__":
    main()