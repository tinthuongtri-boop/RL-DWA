#!/usr/bin/env python3
"""
test_mobile_episode_dryrun.py — Chay 1 episode ngan voi robot mobile,
KHONG can SAC (dung DWA weight co dinh), de kiem tra hanh vi runtime
truoc khi bat dau train.

Kiem tra chinh:
  1. env.reset() khong bi loi (sample duoc pose khong va cham).
  2. Ngay step dau tien, khi robot con dung im tai spawn, min_lidar phai
     LON (khong bi tu bat trung than robot). Neu min_lidar rat nho ngay
     tu step 0 o mot world trong, day la bang chung fix geomgroup trong
     sensor_utils.py CHUA duoc ap dung dung.
  3. Robot co di chuyen theo lenh DWA hay khong (x,y thay doi qua cac step).

Chay tu folder SAC_Sparse_HER, SAU KHI da:
  - dat file gym_bunker_env_mobile.py (ban da fix) o project root
  - dat file sensor_utils.py (ban da fix) o project root, GHI DE file cu
  - dam bao lib/bunker_velocity_controller_mobile.py da co san

Usage:
    python3 test_mobile_episode_dryrun.py ../assets/worlds/empty.xml
    python3 test_mobile_episode_dryrun.py ../assets/worlds/test/world_16_hard.xml --steps 100 --render
"""
import os
import sys
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Fixed DWA weights (alpha=heading, beta=clearance, gamma=velocity).
# Khong can SAC cho bai test nay -- muc dich la kiem tra sensor + controller,
# khong phai kiem tra policy.
FIXED_ACTION = np.array([0.5, 0.3, 0.2], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    # Dat mac dinh tro ra folder assets ngoai SAC_Sparse_HER qua ../assets/worlds/empty.xml
    p.add_argument("world", nargs="?", default="../assets/worlds/empty.xml",
                   help="Duong dan world xml, vd ../assets/worlds/empty.xml")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--render", action="store_true")
    p.add_argument("--self-hit-threshold", type=float, default=1.0,
                   help="Nguong (m): min_lidar duoi muc nay NGAY TU STEP 0 "
                        "(khi robot chua kip cham vat can nao) bi coi la "
                        "dau hieu tu bat trung than robot.")
    return p.parse_args()


def main():
    args = parse_args()
    xml_path = os.path.abspath(args.world)
    if not os.path.exists(xml_path):
        print(f"[FAIL] World khong ton tai: {xml_path}")
        sys.exit(1)

    try:
        from gym_bunker_env_mobile import BunkerEnv
    except Exception as e:
        print(f"[FAIL] Khong import duoc BunkerEnv tu gym_bunker_env_mobile.py: {e}")
        print("       Kiem tra file da dat dung vi tri (project root) chua.")
        sys.exit(1)

    render_mode = "human" if args.render else None
    env = BunkerEnv(xml_path=xml_path, render_mode=render_mode,
                    max_goal_sampling_distance=12.0)

    print(f"\n{'=' * 70}\nDRY RUN: {xml_path}\n{'=' * 70}")
    obs, info = env.reset(seed=args.seed)
    start = obs["achieved_goal"][:2].copy()
    goal  = obs["desired_goal"][:2].copy()
    print(f"[OK] reset() thanh cong. start={start.round(2)}  goal={goal.round(2)}")

    ok = True
    first_min_lidar = None
    positions = []

    for step in range(args.steps):
        obs, reward, terminated, truncated, step_info = env.step(FIXED_ACTION)
        min_r = float(step_info.get("min_lidar", np.nan))
        pos   = obs["achieved_goal"][:2].copy()
        positions.append(pos)

        if step == 0:
            first_min_lidar = min_r

        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step={step:3d}  pos=({pos[0]:+6.2f},{pos[1]:+6.2f})  "
                  f"min_lidar={min_r:6.2f}  reward={reward:+7.2f}  "
                  f"collision={step_info.get('collision')}")

        if terminated or truncated:
            print(f"[INFO] Episode ket thuc som o step {step} "
                  f"(collision={step_info.get('collision')}, "
                  f"success={step_info.get('is_success')}).")
            break

    env.close()

    print(f"\n{'-' * 70}\nDANH GIA\n{'-' * 70}")

    # Check 1: self-hit ngay step dau tien
    if first_min_lidar is not None and first_min_lidar < args.self_hit_threshold:
        print(f"[WARN] min_lidar o step 0 = {first_min_lidar:.3f}m "
              f"(< nguong {args.self_hit_threshold}m). Neu world nay khong co "
              f"vat can ngay sat spawn, day la dau hieu LiDAR dang tu bat trung "
              f"than robot -> kiem tra lai fix geomgroup trong sensor_utils.py "
              f"co duoc ap dung dung khong (group=2 tren toan bo geom cua robot "
              f"trong mobile_body.xml, va geomgroup duoc truyen vao mj_ray).")
        ok = False
    else:
        print(f"[OK] min_lidar o step 0 = {first_min_lidar:.3f}m -- hop ly, "
              f"khong co dau hieu tu bat trung than robot.")

    # Check 2: robot co di chuyen khong (track width dung -> DWA dieu khien duoc)
    positions = np.array(positions)
    if len(positions) >= 2:
        total_disp = float(np.linalg.norm(positions[-1] - positions[0]))
        print(f"[INFO] Tong dich chuyen sau {len(positions)} step: {total_disp:.3f}m")
        if total_disp < 1e-3:
            print("[WARN] Robot hau nhu khong di chuyen -- co the do DWA khong tim "
                  "duoc trajectory kha thi (vd robot_radius qua lon so voi map), "
                  "hoac velocity controller khong nhan lenh dung.")

    print(f"\n{'PASS' if ok else 'CHECK WARN'}: {xml_path}")


if __name__ == "__main__":
    main()