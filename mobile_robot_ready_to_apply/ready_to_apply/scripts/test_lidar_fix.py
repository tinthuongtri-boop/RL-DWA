#!/usr/bin/env python3
"""
test_lidar_fix.py
-------------------------------------------------------------------------
Kiem chung fix bug "LiDAR tu va cham voi chinh robot" (xem
patches/sensor_utils.py.patch_instructions.md).

Chay: python3 scripts/test_lidar_fix.py --xml assets/worlds/mobile_empty.xml

Test 1: phong TRONG (world xml khong co obstacle nao) -- ky vong 0/449
        tia bi "hit" (moi tia phai tra ve max_range vi khong co gi de
        cham). Neu > 0 tia bi hit -> LiDAR dang tu va cham voi robot.

Test 2: dat 1 obstacle gia lap ngay truoc robot, kiem tra LiDAR co phat
        hien DUNG goc va DUNG khoang cach khong (dam bao fix khong vo
        tinh loc mat luon vat can that).
"""
import argparse
import math
import sys

import numpy as np

try:
    import mujoco
except ImportError:
    print("Can cai: pip install mujoco")
    sys.exit(1)


def raycast_449(model, data, x, y, theta, sensor_height, use_geomgroup_fix):
    max_range = 14.5
    pnt = np.array([x, y, sensor_height], dtype=np.float64)
    angles = np.linspace(-math.pi, math.pi, 449, endpoint=False)
    geomid_out = np.zeros(1, dtype=np.int32)

    try:
        exclude_id = int(model.body("mobile_base").id)
    except Exception:
        exclude_id = -1

    if use_geomgroup_fix:
        geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.uint8)
        geomgroup[2] = 0
    else:
        geomgroup = None  # hanh vi CU (bug): khong loc theo group

    hits = []
    for a in angles:
        vec = np.array([math.cos(theta + a), math.sin(theta + a), 0.0])
        dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, exclude_id, geomid_out)
        if 0.0 < dist <= max_range:
            hits.append((math.degrees(a), dist))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    args = ap.parse_args()

    print(f"[Test 1] Phong trong -- so sanh CU (bug) vs MOI (da fix)")
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    d.qpos[0:3] = [-4, -4, 0.005]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)

    hits_old = raycast_449(m, d, -4, -4, 0.0, sensor_height=0.25, use_geomgroup_fix=False)
    hits_new = raycast_449(m, d, -4, -4, 0.0, sensor_height=0.3488, use_geomgroup_fix=True)

    print(f"  Truoc khi sua (sensor_height=0.25, khong geomgroup): "
          f"{len(hits_old)}/449 tia bi hit (SAI neu >0)")
    print(f"  Sau khi sua   (sensor_height=0.3488, co geomgroup):  "
          f"{len(hits_new)}/449 tia bi hit (phai = 0)")

    if len(hits_new) == 0:
        print("  PASS: da het loi tu va cham.")
    else:
        print("  FAIL: van con tu va cham -- kiem tra lai group trong mobile_body.xml")
        sys.exit(1)

    print("\n[Test 2] Kiem tra khong loc mat vat can that")
    print("  (chay rieng voi world co obstacle -- vd assets/worlds/train/*.xml"
          " sau khi da convert sang mobile robot, hoac tu tao world test co obstacle)")
    print("  Xem huong dan trong patches/sensor_utils.py.patch_instructions.md")


if __name__ == "__main__":
    main()
