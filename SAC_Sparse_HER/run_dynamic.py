import mujoco
import mujoco.viewer
import numpy as np
import time

# LOAD MODEL
model = mujoco.MjModel.from_xml_path(
    "dynamic_single.xml"
)

data = mujoco.MjData(model)

# VIEWER
viewer = mujoco.viewer.launch_passive(model, data)

print("Simulation started...")

while viewer.is_running():

    t = data.time

    # =================================================
    # ZONE 3 : SLOW OBSTACLE
    # =================================================

    data.ctrl[0] = 3.5 * np.sin(0.05 * t)

    # =================================================
    # ZONE 4 : MULTI DYNAMIC
    # =================================================

    data.ctrl[1] = 3.5 * np.sin(0.08 * t)

    data.ctrl[2] = 3.5 * np.sin(0.1 * t + 1.5)

    # =================================================
    # ZONE 6 : CROSSING OBSTACLES
    # =================================================

    data.ctrl[3] = 3.5 * np.sin(0.05 * t)

    data.ctrl[4] = 55 + 2 * np.sin(0.05 * t)

    # STEP
    mujoco.mj_step(model, data)

    viewer.sync()

    time.sleep(0.001)

viewer.close()