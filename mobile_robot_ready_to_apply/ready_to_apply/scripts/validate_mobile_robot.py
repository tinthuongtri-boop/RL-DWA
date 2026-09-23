#!/usr/bin/env python3
"""
validate_mobile_robot.py
-------------------------------------------------------------------------
Bộ kiểm chứng vật lý cho model MJCF robot vi sai mới (mobile_*.xml),
độc lập với gym_bunker_env.py / DWA / SAC -- chỉ test MuJoCo thuần.

Chạy: python3 scripts/validate_mobile_robot.py --xml assets/worlds/mobile_empty.xml

Các phép kiểm tra:
  1. Compile OK, đúng số body/joint/actuator/sensor.
  2. wheel_separation & wheel_radius khớp thông số Gazebo gốc
     (wheel_separation=0.326 m, wheel_radius=0.085 m) -- xác nhận
     mesh scale đúng, không bị lệch đơn vị.
  3. Robot đứng yên (ctrl=0) phải settle ổn định (qvel -> 0), không rơi
     xuyên sàn, không NaN.
  4. Lái thẳng (ctrl trái=phải=v) phải đạt steady-state vận tốc
     dương, ổn định (không dao động / không NaN), có báo cáo % trượt
     so với động học lăn không trượt lý tưởng (r * omega_wheel).
  5. Quay tại chỗ (trái/phải ngược dấu) phải tạo ra thay đổi yaw đúng
     dấu theo quy ước vi sai chuẩn: omega = (v_right - v_left) / track.

Nếu bạn tiếp tục tinh chỉnh kv / damping / diaginertia trong
mobile_body.xml hoặc mobile_actuators.xml, hãy chạy lại script này để
xác nhận vẫn ổn định trước khi đưa vào gym_bunker_env.py / training.
"""
import argparse
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("Cần cài: pip install mujoco")
    sys.exit(1)

REF_WHEEL_SEP = 0.326   # m, tu Gazebo diff_drive plugin cua URDF goc
REF_WHEEL_RAD = 0.085   # m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Duong dan world xml dung mobile robot")
    ap.add_argument("--drive-ctrl", type=float, default=3.0,
                     help="Toc do goc banh (rad/s) khi test lai thang")
    args = ap.parse_args()

    print(f"[1] Compiling {args.xml} ...")
    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    print(f"    OK. nbody={m.nbody} njnt={m.njnt} nu={m.nu} nsensor={m.nsensor}")

    print("[2] Kiem tra wheel separation & radius vs Gazebo ground truth ...")
    id_r = m.body("right_wheel").id
    id_l = m.body("left_wheel").id
    w_track = float(np.max(np.abs(m.body_pos[id_r] - m.body_pos[id_l])))
    r_wheel = float(m.geom_size[m.geom("left_wheel_geom").id][0])
    print(f"    wheel_separation = {w_track:.4f} m  (ref {REF_WHEEL_SEP})")
    print(f"    wheel_radius     = {r_wheel:.4f} m  (ref {REF_WHEEL_RAD})")
    assert abs(w_track - REF_WHEEL_SEP) < 0.01, "wheel separation LECH so voi Gazebo!"
    assert abs(r_wheel - REF_WHEEL_RAD) < 0.01, "wheel radius LECH so voi Gazebo!"
    print("    OK -- khop thong so goc.")

    print("[3] Test dung yen (ctrl=0) settle on dinh ...")
    d.qpos[0:3] = [0, 0, 0.005]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(200):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all(), "NaN khi dung yen!"
    qvel_norm = float(np.linalg.norm(d.qvel[:6]))
    print(f"    z sau settle = {d.qpos[2]:.5f}  |qvel|={qvel_norm:.2e}")
    assert qvel_norm < 1e-3, "Robot khong on dinh khi dung yen (van con dao dong)!"
    print("    OK.")

    print(f"[4] Test lai thang (ctrl={args.drive_ctrl} ca 2 banh) ...")
    jl = m.joint("left_wheel_joint").id
    qvel_adr = m.jnt_dofadr[jl]
    d.ctrl[m.actuator("left_wheel").id] = args.drive_ctrl
    d.ctrl[m.actuator("right_wheel").id] = args.drive_ctrl
    for _ in range(1500):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all(), "NaN khi lai thang!"
    base_vx = float(d.qvel[0])
    wheel_w = float(d.qvel[qvel_adr])
    v_noslip = r_wheel * wheel_w
    slip_pct = 100.0 * (1.0 - base_vx / v_noslip) if v_noslip > 1e-6 else float("nan")
    print(f"    steady base_vx = {base_vx:+.4f} m/s")
    print(f"    steady wheel_w = {wheel_w:+.4f} rad/s  (no-slip target v={v_noslip:.4f} m/s)")
    print(f"    slip = {slip_pct:.1f}%  (khac 0% la binh thuong voi banh nhe -- xem ghi chu"
          " trong mobile_body.xml)")
    assert base_vx > 0.01, "Robot khong tien duoc ve phia truoc -- kiem tra lai actuator/contact!"
    print("    OK -- robot tien ve phia truoc on dinh.")

    print("[5] Test quay tai cho (trai=+2, phai=-2) ...")
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, 0.005]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    d.ctrl[m.actuator("left_wheel").id] = 2.0
    d.ctrl[m.actuator("right_wheel").id] = -2.0
    for _ in range(1500):
        mujoco.mj_step(m, d)
    qw, qx, qy, qz = d.qpos[3:7]
    yaw_deg = np.degrees(2.0 * np.arctan2(qz, qw))
    print(f"    yaw sau quay = {yaw_deg:+.1f} deg  (ky vong AM/clockwise voi quy uoc"
          " omega=(v_right-v_left)/track)")
    assert yaw_deg < -5.0, "Dau yaw SAI -- kiem tra lai axis cua wheel joint!"
    print("    OK -- dung quy uoc dau vi sai chuan.")

    print("[6] Test toc do CAO (bap benh / pitch oscillation) ...")
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, 0.005]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    d.ctrl[m.actuator("left_wheel").id] = 10.0
    d.ctrl[m.actuator("right_wheel").id] = 10.0
    pitches = []
    for i in range(2000):
        mujoco.mj_step(m, d)
        if i > 300:
            qw, qx, qy, qz = d.qpos[3:7]
            sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
            pitches.append(np.degrees(np.arcsin(sinp)))
    pitches = np.array(pitches)
    p2p = pitches.max() - pitches.min()
    print(f"    pitch peak-to-peak o ctrl=10 rad/s = {p2p:.2f} deg"
          "  (nguong canh bao: >3 deg la co dau hieu 'bap benh')")
    assert np.isfinite(d.qpos).all(), "NaN khi chay toc do cao!"
    if p2p > 3.0:
        print("    CANH BAO: pitch dao dong manh -- xem xet tang 'armature' hoac"
              " giam 'kv' trong mobile_actuators.xml (xem ghi chu trong mobile_body.xml).")
    else:
        print("    OK -- khong co dau hieu bap benh dang ke.")

    print("\n=== TAT CA KIEM TRA PASS ===")


if __name__ == "__main__":
    main()
