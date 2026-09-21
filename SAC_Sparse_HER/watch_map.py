import mujoco
import mujoco.viewer
import numpy as np
import time

# ====================== LOAD MODEL ======================
model = mujoco.MjModel.from_xml_path("dynamic_single.xml")  # Đổi tên file nếu khác
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

# Lấy actuator ID
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_y_ctrl")

print(f"Số actuator: {model.nu}")
if act_id != -1:
    print("✅ Tìm thấy actuator: move_y_ctrl")
else:
    print("❌ Không tìm thấy actuator move_y_ctrl")

print("\nSimulation started! Press ESC to exit.\n")

while viewer.is_running():
    t = data.time
    
    # Điều khiển vật cản động (di chuyển ngang qua lại)
    if act_id != -1:
        data.ctrl[act_id] = 4.0 * np.sin(t * 1.2)   # Thay 1.2 để thay tốc độ

    mujoco.mj_step(model, data)
    viewer.sync()
    
    time.sleep(0.001)  # Giới hạn FPS

viewer.close()
print("Simulation ended.")