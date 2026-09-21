#!/usr/bin/env python3
"""
run_dwa_sac.py — Chạy DWA-SAC hybrid với model đã train
---------------------------------------------------------
Dùng SEED cố định giống run_dwa_only.py → cùng start/goal.
SAC xuất ra (α, β, γ) động theo context.
"""

import os
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from comparison_common import (
    RUN_NAME, MAX_GOAL_DIST, MAX_EP_STEPS,
    get_xml_path, get_seeds,
    run_single_episode,
    run_single_episode_with_astar,
    save_results, 
    print_summary,
    _SCRIPT_DIR,
)
from gym_bunker_env import BunkerEnv



def main():
    xml  = get_xml_path()
    ckpt = os.path.join(_SCRIPT_DIR, "log", RUN_NAME,
                        "best_model", "best_model.zip")

    print(f"╔{'═'*58}╗")
    print(f"║  RUN 2: DWA + SAC (hybrid)                               ║")
    print(f"║  Model: {RUN_NAME:<49s}║")
    print(f"╚{'═'*58}╝\n")

    # ── Tạo env trước ─────────────────────────────────────────────
    env = TimeLimit(
        BunkerEnv(xml_path=xml, render_mode= "human",
                  max_goal_sampling_distance=MAX_GOAL_DIST),
        max_episode_steps=MAX_EP_STEPS,
    )

    # ── Wrap env thành DummyVecEnv để SAC.load chấp nhận ──────────
    # HerReplayBuffer cần env → ta dùng env đã có sẵn (không tạo thêm
    # MuJoCo mới, tránh conflict mjcb_control callback toàn cục)
    _called = [False]
    def _factory():
        if _called[0]:
            raise RuntimeError("Factory chỉ gọi được 1 lần")
        _called[0] = True
        return env

    vec_env   = DummyVecEnv([_factory])
    print(f"📂 Loading SAC model: {ckpt}")
    sac_model = SAC.load(ckpt, env=vec_env, device="auto")
    print(f"✓  Model loaded\n")

    # ── Policy: SAC predict ────────────────────────────────────────
    def sac_policy(obs):
        # obs là dict từ TimeLimit (không batched) → add batch dim
        obs_batched = {k: np.array(v)[np.newaxis, :] for k, v in obs.items()}
        action, _ = sac_model.predict(obs_batched, deterministic=True)
        return action[0].astype(np.float32)

    # ── Chạy các episodes với seeds cố định ────────────────────────
    seeds   = get_seeds()
    results = []

    for i, seed in enumerate(seeds):
        # Dùng env.reset(seed=seed) qua unwrap vì DummyVecEnv chỉ cós
        # env.seed() + env.reset() tách biệt. Ở đây ta access env
        # trực tiếp (TimeLimit) thay vì qua vec_env.
        r = run_single_episode(env, sac_policy, seed=seed)
        results.append(r)

        # Log thêm mean weights SAC chọn trong ep này
        w = np.mean(r["weights"], axis=0) if r["weights"] else [0,0,0]
        print(f"  EP {i+1:2d}/{len(seeds)}  seed={seed}  "
              f"{r['outcome']:10s}  {r['steps']:3d} steps  "
              f"dist={r['init_dist']:4.1f}m  SPL={r['spl']:.2f}  "
              f"│  α={w[0]:.2f} β={w[1]:.2f} γ={w[2]:.2f}")

    env.close()

    save_results(results, method_name="dwa_sac",
                 extra_meta={"run_name": RUN_NAME})

    print_summary("DWA + SAC", results)


if __name__ == "__main__":
    main()