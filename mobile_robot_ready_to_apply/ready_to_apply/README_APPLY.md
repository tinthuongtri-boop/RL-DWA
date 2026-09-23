# Cách áp dụng gói này vào repo rl_nav/

1. Copy 3 file đã patch sẵn đè lên bản gốc trong repo:
   - gym_bunker_env.py        -> rl_nav/gym_bunker_env.py
   - sensor_utils.py          -> rl_nav/sensor_utils.py
   - dwa_standalone_demo.py   -> rl_nav/dwa_standalone_demo.py

2. Copy toàn bộ thư mục assets/mobile/ vào rl_nav/assets/mobile/
   (ngang hàng với assets/bunker/ hiện có).

3. Copy assets/worlds/mobile_empty.xml vào rl_nav/assets/worlds/mobile_empty.xml

4. Từ ROOT của repo (rl_nav/), chạy:
       chmod +x scripts/convert_world_files.sh
       ./scripts/convert_world_files.sh
   rồi review bằng: git diff -- assets/worlds/ | less
   (script này sẽ tự sửa toàn bộ 40+ world xml train/test/val của bạn
   sang include mobile_*.xml thay vì bunker_*.xml, và đổi
   integrator="RK4" -> "implicitfast")

5. Validate model vật lý trước khi động vào Python:
       pip install mujoco --break-system-packages
       python3 scripts/validate_mobile_robot.py --xml assets/worlds/mobile_empty.xml
       python3 scripts/test_lidar_fix.py --xml assets/worlds/mobile_empty.xml

6. Test DWA thuần (chưa SAC) trên world thật, ví dụ World_Hard:
       python3 dwa_standalone_demo.py --xml assets/worlds/val/World_Hard.xml --auto-start

7. Nếu bước 6 ổn -> train lại SAC từ đầu (best_model.zip cũ của Bunker
   KHÔNG dùng lại được, xem lý do trong docs/INTEGRATION_GUIDE.md mục 10).
