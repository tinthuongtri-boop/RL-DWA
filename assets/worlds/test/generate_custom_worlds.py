#!/usr/bin/env python3
"""
generate_custom_worlds.py
-------------------------
Sinh 4 file MuJoCo XML với các mức độ thử thách khác nhau:
  - custom_easy.xml       : 10 obstacles, khoảng cách rộng
  - custom_medium.xml     : 18 obstacles, bắt đầu có hành lang hẹp
  - custom_hard.xml       : 28 obstacles, nhiều ngõ cụt + chokepoint
  - custom_very_hard.xml  : 22 static + 6 DYNAMIC obstacles (di chuyển qua lại)

World bounds: [-4, 10] x [-4, 10] — đồng bộ với env config.
Robot bắt đầu / Goal sampled ngẫu nhiên — obstacles phải chừa vùng "spawn zones".

Chạy:
    python3 generate_custom_worlds.py
Output:
    ./custom_easy.xml
    ./custom_medium.xml
    ./custom_hard.xml
    ./custom_very_hard.xml
"""

import os


# ── XML template ─────────────────────────────────────────────────────
HEADER = """<mujoco model="Agilex Bunker">
  <compiler angle="radian" meshdir="../../bunker" coordinate="local" autolimits="true"/>
  <option timestep="0.01" cone="elliptic" integrator="RK4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <default>
    <default class="agilex_bunker"></default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <include file="../../bunker/bunker_assests.xml"/>
  </asset>

  <worldbody>
    <light name="spotlight" mode="targetbodycom" target="mobile_base" pos="0 0 100" castshadow="false"/>
    <geom conaffinity="1" condim="3" name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <include file="../../bunker/bunker_body.xml"/>

    <body name="mid_goal_marker" pos="-2 -4 0.1" mocap="true">
      <site name="mid_goal_sphere" type="sphere" size="0.1" rgba="1.0 1.0 0.0 1"/>
    </body>

    <body name="final_goal_marker" pos="-2 -4 0.1" mocap="true">
      <site name="final_goal_sphere" type="sphere" size="0.1" rgba="1.0 0.0 0.0 1"/>
    </body>
"""

FOOTER = """  </worldbody>

  <include file="../../bunker/bunker_actuators.xml"/>

</mujoco>
"""


# ── Helper: obstacle tag builders ────────────────────────────────────
def box_obstacle(idx, x, y, sx, sy, color="0.6 0.2 0 1", z=None):
    """Box tĩnh — height = sz."""
    sz = 0.30 if z is None else z
    return f"""
    <body name="obstacle_{idx}" pos="{x:.3f} {y:.3f} {sz:.3f}">
      <geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{color}" contype="1" conaffinity="1"/>
    </body>
"""


def cyl_obstacle(idx, x, y, r, color="0 0.6 0.2 1", z=None):
    """Cylinder tĩnh."""
    sz = 0.30 if z is None else z
    return f"""
    <body name="obstacle_{idx}" pos="{x:.3f} {y:.3f} {sz:.3f}">
      <geom type="cylinder" size="{r:.3f} {sz:.3f}" rgba="{color}" contype="1" conaffinity="1"/>
    </body>
"""


def moving_obstacle(idx, x_start, y_start, x_end, y_end, r=0.3, period=8.0):
    """
    Vật cản động: di chuyển qua lại giữa 2 điểm bằng slide joint + sine motion.
    Dùng MuJoCo's slide joint với ctrl = sin(t) để tạo chuyển động qua lại.

    Args:
      period : chu kỳ 1 lần qua-lại (giây)
      r      : bán kính cylinder
    """
    # Vector từ start → end
    dx = x_end - x_start
    dy = y_end - y_start
    dist = (dx**2 + dy**2)**0.5
    if dist < 1e-6:
        ax, ay = 1.0, 0.0
        dist = 1.0
    else:
        ax = dx / dist
        ay = dy / dist

    # Slide axis vector (normalized direction start→end)
    mid_x = (x_start + x_end) / 2
    mid_y = (y_start + y_end) / 2
    half_range = dist / 2

    return f"""
    <body name="obstacle_dyn_{idx}" pos="{mid_x:.3f} {mid_y:.3f} 0.30">
      <joint name="dyn_slide_{idx}" type="slide" axis="{ax:.3f} {ay:.3f} 0"
             range="-{half_range:.3f} {half_range:.3f}"/>
      <geom type="cylinder" size="{r:.3f} 0.30" rgba="0.9 0.1 0.1 1"
            contype="1" conaffinity="1" mass="10"/>
    </body>
"""


# ══════════════════════════════════════════════════════════════════════
# MAP 1: EASY  —  10 obstacles, thưa, nhiều đường thoáng
# ══════════════════════════════════════════════════════════════════════
def make_easy():
    obs = []
    i = 0
    # Khu vực 1: góc trên-phải, 3 vật cản rời rạc
    obs.append(cyl_obstacle(i, 6.5, 8.0, 0.35));         i += 1
    obs.append(box_obstacle(i, 5.0, 6.5, 0.40, 0.25));   i += 1
    obs.append(cyl_obstacle(i, 7.5, 5.0, 0.30));         i += 1

    # Khu vực 2: giữa map, 3 vật cản
    obs.append(box_obstacle(i, 2.5, 3.0, 0.35, 0.30));   i += 1
    obs.append(cyl_obstacle(i, 4.0, 1.5, 0.25));         i += 1
    obs.append(box_obstacle(i, 0.5, 4.5, 0.30, 0.40));   i += 1

    # Khu vực 3: góc dưới-trái, 2 vật cản
    obs.append(cyl_obstacle(i, -2.5, -1.5, 0.35));       i += 1
    obs.append(box_obstacle(i, -1.0, -3.0, 0.40, 0.30)); i += 1

    # Khu vực 4: góc trên-trái + dưới-phải
    obs.append(box_obstacle(i, -2.5, 6.0, 0.30, 0.35));  i += 1
    obs.append(cyl_obstacle(i, 7.5, -2.0, 0.30));        i += 1

    return "".join(obs), "EASY  —  10 obstacles, thưa, nhiều corridor rộng"


# ══════════════════════════════════════════════════════════════════════
# MAP 2: MEDIUM  —  18 obstacles, có hành lang hẹp
# ══════════════════════════════════════════════════════════════════════
def make_medium():
    obs = []
    i = 0

    # Cluster A: góc trên-phải — tạo "phòng" với lối vào hẹp
    obs.append(box_obstacle(i, 7.0, 7.5, 1.20, 0.20));   i += 1   # tường ngang dài
    obs.append(box_obstacle(i, 8.5, 6.0, 0.20, 1.20));   i += 1   # tường dọc dài
    obs.append(cyl_obstacle(i, 6.0, 5.5, 0.30));         i += 1
    obs.append(box_obstacle(i, 5.5, 8.0, 0.30, 0.25));   i += 1

    # Cluster B: giữa map — slalom xuyên qua
    obs.append(cyl_obstacle(i, 3.0, 4.0, 0.35));         i += 1
    obs.append(cyl_obstacle(i, 4.5, 2.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 3.5, 1.0, 0.30));         i += 1
    obs.append(box_obstacle(i, 1.5, 2.5, 0.35, 0.30));   i += 1

    # Cluster C: hành lang hẹp dọc
    obs.append(box_obstacle(i, -1.0, 4.0, 0.15, 0.80));  i += 1   # tường ngăn
    obs.append(box_obstacle(i, 0.5, 4.0, 0.15, 0.80));   i += 1   # tường ngăn
    # → tạo corridor rộng 1.5m giữa 2 tường

    # Cluster D: góc dưới-trái
    obs.append(cyl_obstacle(i, -2.5, -2.0, 0.40));       i += 1
    obs.append(box_obstacle(i, -1.5, -1.0, 0.30, 0.25)); i += 1
    obs.append(box_obstacle(i, -3.0, 1.0, 0.25, 0.35));  i += 1

    # Cluster E: góc trên-trái
    obs.append(box_obstacle(i, -2.5, 7.0, 0.40, 0.25));  i += 1
    obs.append(cyl_obstacle(i, -1.0, 8.0, 0.30));        i += 1

    # Rải rác
    obs.append(cyl_obstacle(i, 8.0, 2.0, 0.30));         i += 1
    obs.append(box_obstacle(i, 6.5, -1.5, 0.30, 0.40));  i += 1
    obs.append(cyl_obstacle(i, 5.0, -2.5, 0.25));        i += 1

    return "".join(obs), "MEDIUM  —  18 obstacles, có hành lang hẹp 1.5m"


# ══════════════════════════════════════════════════════════════════════
# MAP 3: HARD  —  28 obstacles, nhiều ngõ cụt + chokepoint
# ══════════════════════════════════════════════════════════════════════
def make_hard():
    obs = []
    i = 0

    # Cluster A: "Phòng" ở góc trên-phải với lối vào duy nhất rộng 1.0m
    obs.append(box_obstacle(i, 6.5, 8.0, 1.50, 0.15));   i += 1   # tường trên
    obs.append(box_obstacle(i, 8.0, 6.5, 0.15, 1.50));   i += 1   # tường phải
    obs.append(box_obstacle(i, 5.0, 6.5, 0.15, 1.0));    i += 1   # tường trái (có khe)
    # → khe vào từ dưới ở y = 5.0

    # Bên trong phòng: có obstacle cản giữa
    obs.append(cyl_obstacle(i, 6.8, 7.0, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 6.0, 7.8, 0.25));         i += 1

    # Cluster B: slalom dày đặc giữa map
    obs.append(cyl_obstacle(i, 2.5, 5.5, 0.35));         i += 1
    obs.append(cyl_obstacle(i, 4.0, 4.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 2.5, 3.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 4.0, 2.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 2.5, 1.5, 0.30));         i += 1

    # Cluster C: tường dài tạo chokepoint
    obs.append(box_obstacle(i,  0.0, 0.5, 1.20, 0.15));  i += 1   # tường dài
    obs.append(box_obstacle(i, -1.5, 2.5, 0.15, 1.00));  i += 1   # tường dọc
    # → chokepoint rộng ~1.0m

    # Cluster D: ngõ cụt dụ trap
    obs.append(box_obstacle(i, -2.5, 4.5, 0.15, 0.80));  i += 1
    obs.append(box_obstacle(i, -2.5, 5.8, 0.15, 0.40));  i += 1
    obs.append(box_obstacle(i, -1.5, 5.2, 0.80, 0.15));  i += 1
    # → ngõ cụt, SAC phải học back-out

    # Cluster E: obstacle dày ở góc dưới
    obs.append(cyl_obstacle(i, -2.5, -2.5, 0.40));       i += 1
    obs.append(box_obstacle(i, -1.0, -2.0, 0.30, 0.30)); i += 1
    obs.append(box_obstacle(i,  0.5, -3.0, 0.35, 0.25)); i += 1
    obs.append(cyl_obstacle(i,  2.0, -2.0, 0.30));       i += 1
    obs.append(cyl_obstacle(i,  3.5, -2.5, 0.30));       i += 1

    # Cluster F: góc trên-trái dày
    obs.append(box_obstacle(i, -3.0, 7.5, 0.40, 0.25));  i += 1
    obs.append(box_obstacle(i, -1.5, 8.0, 0.25, 0.40));  i += 1
    obs.append(cyl_obstacle(i, -2.5, 6.5, 0.25));        i += 1

    # Rải rác
    obs.append(cyl_obstacle(i, 8.5, 3.0, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 7.0, 1.5, 0.25));         i += 1
    obs.append(box_obstacle(i, 8.5, -1.0, 0.25, 0.35));  i += 1
    obs.append(cyl_obstacle(i, 5.5, -1.5, 0.30));        i += 1
    obs.append(cyl_obstacle(i, 4.0, 7.5, 0.25));         i += 1

    return "".join(obs), "HARD  —  28 obstacles, ngõ cụt + chokepoint 1.0m"


# ══════════════════════════════════════════════════════════════════════
# MAP 4: VERY HARD  —  22 static + 6 DYNAMIC obstacles
# ══════════════════════════════════════════════════════════════════════
def make_very_hard():
    obs = []
    actuator_tags = []
    i = 0

    # ── Static obstacles: giống hard nhưng bớt đi 6 để chừa chỗ cho dynamic ──

    # Tường khung ngoài (chống robot thoát ra rìa)
    obs.append(box_obstacle(i, 6.5, 8.5, 1.50, 0.15));   i += 1
    obs.append(box_obstacle(i, 8.5, 6.0, 0.15, 1.50));   i += 1

    # Slalom giữa map
    obs.append(cyl_obstacle(i, 2.5, 5.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 4.0, 4.5, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 2.5, 3.5, 0.30));         i += 1

    # Chokepoint
    obs.append(box_obstacle(i,  0.0, 0.5, 1.00, 0.15));  i += 1
    obs.append(box_obstacle(i, -1.5, 2.0, 0.15, 0.80));  i += 1

    # Ngõ cụt
    obs.append(box_obstacle(i, -2.5, 4.5, 0.15, 0.80));  i += 1
    obs.append(box_obstacle(i, -1.5, 5.2, 0.80, 0.15));  i += 1

    # Góc dưới-trái
    obs.append(cyl_obstacle(i, -2.5, -2.5, 0.40));       i += 1
    obs.append(box_obstacle(i, -1.0, -2.0, 0.30, 0.30)); i += 1
    obs.append(cyl_obstacle(i,  2.0, -2.0, 0.30));       i += 1

    # Góc trên-trái
    obs.append(box_obstacle(i, -3.0, 7.5, 0.40, 0.25));  i += 1
    obs.append(cyl_obstacle(i, -2.5, 6.5, 0.25));        i += 1

    # Rải rác
    obs.append(cyl_obstacle(i, 8.5, 3.0, 0.30));         i += 1
    obs.append(cyl_obstacle(i, 7.0, 1.5, 0.25));         i += 1
    obs.append(box_obstacle(i, 8.5, -1.0, 0.25, 0.35));  i += 1
    obs.append(cyl_obstacle(i, 5.5, -1.5, 0.30));        i += 1
    obs.append(cyl_obstacle(i, 4.0, 7.5, 0.25));         i += 1
    obs.append(box_obstacle(i, 6.5, 7.0, 0.30, 0.30));   i += 1
    obs.append(cyl_obstacle(i, 5.0, 2.0, 0.25));         i += 1
    obs.append(box_obstacle(i, 1.0, -1.0, 0.25, 0.30));  i += 1

    # ── DYNAMIC obstacles: 6 cái di chuyển qua lại ───────────────────
    # Mỗi cái có 1 slide joint + actuator sine motion
    dyn_configs = [
        # (start_x, start_y, end_x, end_y, radius, period)
        (0.0, 6.0,   2.5, 6.0,  0.30, 8.0),   # dyn_0: di chuyển ngang trên cao
        (5.5, 4.5,   5.5, 6.5,  0.30, 10.0),  # dyn_1: di chuyển dọc giữa
        (0.5, -1.5,  3.0, -1.5, 0.25, 6.0),   # dyn_2: di chuyển ngang phía dưới
        (7.5, 5.0,   7.5, 7.0,  0.25, 12.0),  # dyn_3: di chuyển dọc phải
        (-1.0, 3.0, -1.0, 1.0,  0.25, 7.0),   # dyn_4: di chuyển dọc trái (gần chokepoint)
        (3.0, 7.0,   5.0, 7.0,  0.30, 9.0),   # dyn_5: di chuyển ngang trên
    ]

    for k, (xs, ys, xe, ye, r, T) in enumerate(dyn_configs):
        obs.append(moving_obstacle(k, xs, ys, xe, ye, r=r, period=T))
        # Actuator để tạo motion: dùng position actuator với cyclic ctrl
        # Ctrl sẽ được set trong env bằng hàm thay đổi theo time
        dist = ((xe-xs)**2 + (ye-ys)**2)**0.5 / 2
        actuator_tags.append(f"""    <position name="dyn_act_{k}" joint="dyn_slide_{k}"
              kp="300" kv="50" ctrlrange="-{dist:.3f} {dist:.3f}"/>
""")

    return "".join(obs), "VERY HARD  —  22 static + 6 DYNAMIC obstacles", "".join(actuator_tags)


# ══════════════════════════════════════════════════════════════════════
# WRITE XML FILES
# ══════════════════════════════════════════════════════════════════════
def write_xml(filename: str, obstacles_xml: str, description: str,
              extra_actuators: str = ""):
    """Ghi file XML hoàn chỉnh."""

    # Build comment section
    comment = f"""
    <!-- ═══════════════════════════════════════════════════════════
         CUSTOM WORLD: {description}
         Bounds: [-4, 10] x [-4, 10] meters
         ═══════════════════════════════════════════════════════════ -->
"""

    body_section = HEADER + comment + obstacles_xml + FOOTER

    # Nếu có dynamic actuators, chèn vào trước </mujoco>
    if extra_actuators:
        # Thay thế include actuators default → thêm custom actuators
        body_section = body_section.replace(
            '  <include file="../../bunker/bunker_actuators.xml"/>\n',
            f"""  <include file="../../bunker/bunker_actuators.xml"/>

  <!-- Dynamic obstacle actuators (sinusoidal motion controlled in env) -->
  <actuator>
{extra_actuators}  </actuator>
"""
        )

    with open(filename, "w") as f:
        f.write(body_section)
    print(f"✓ Created: {filename}  ({description})")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Map 1
    obs, desc = make_easy()
    write_xml(os.path.join(out_dir, "custom_easy.xml"), obs, desc)

    # Map 2
    obs, desc = make_medium()
    write_xml(os.path.join(out_dir, "custom_medium.xml"), obs, desc)

    # Map 3
    obs, desc = make_hard()
    write_xml(os.path.join(out_dir, "custom_hard.xml"), obs, desc)

    # Map 4 (có dynamic)
    obs, desc, actuators = make_very_hard()
    write_xml(os.path.join(out_dir, "custom_very_hard.xml"), obs, desc,
              extra_actuators=actuators)

    print("\n" + "═"*60)
    print("Done! 4 custom worlds created.")
    print("═"*60)
    print("""
Copy vào thư mục worlds:
  cp custom_*.xml ~/rl_nav/assets/worlds/test/

Test trong sac_inference.py:
  ENV_NAME = "test/custom_easy"       # hoặc medium/hard/very_hard
  python3 SAC_Sparse_HER/sac_inference.py

Chú ý:
  - very_hard cần code phụ để CONTROL actuator — xem hướng dẫn ở output.
""")
