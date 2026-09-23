# Chuyển robot Bunker (skid-steer 6 bánh) sang robot vi sai (URDF của bạn) trong MuJoCo

## 1. Problem Understanding

Codebase hiện tại (DWA + SAC + HER trên MuJoCo) được xây dựng quanh mô
hình Agilex Bunker: 6 bánh (skid-steer, 3 bánh/bên cùng tốc độ), điều
khiển qua `BunkerVelocityController` (chuyển `(v, ω)` → `(ω_trái, ω_phải)`
bằng công thức vi sai chuẩn rồi phát cùng lệnh cho 3 bánh mỗi bên).

Bạn muốn thay bằng robot vi sai THẬT (2 bánh chủ động + 4 bánh đa hướng
thụ động / caster) từ URDF `urdf_mobile.urdf` + 14 mesh STL đính kèm.

## 2. Current Architecture (điểm chạm với robot model)

```
gym_bunker_env.py
 ├─ _initialize_velocity_controller()  -> đọc body "w_rr"/"w_lr",
 │                                         geom "w_rr_geom", actuator
 │                                         "w_rr,w_rc,w_rf" / "w_lr,w_lc,w_lf"
 ├─ _ROBOT_BODY_PREFIXES               -> lọc self-contact khi check collision
 ├─ _set_qpos_pose(height=0.25)        -> độ cao spawn ban đầu
 └─ get_robot_velocities()             -> đọc body "mobile_base" (KHÔNG đổi)

sensor_utils.py
 └─ get_robot_state_and_lidar()        -> đọc body "mobile_base" (KHÔNG đổi)

dwa_standalone_demo.py
 └─ setup_velocity_controller()        -> giống _initialize_velocity_controller

assets/worlds/**/*.xml (40+ file)
 └─ <include file="../../bunker/bunker_{assests,body,actuators}.xml"/>
```

Điểm mấu chốt: **toàn bộ pipeline quan sát/reward/DWA/SAC/HER không hề
biết gì về hình học robot** — nó chỉ phụ thuộc: (a) tên body gốc
`"mobile_base"`, (b) `n_lidar=449` tia LiDAR, (c) `w_track`/`r_wheel`
được TÍNH TỰ ĐỘNG từ model lúc `reset`, không hard-code số. Do đó chiến
lược chuyển đổi tốt nhất là **giữ nguyên tên body gốc + số tia LiDAR**,
chỉ thay cấu trúc bên trong.

## 3. Evidence (đã kiểm chứng bằng code, không phải suy đoán)

| Việc kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| Đơn vị mesh STL | Mét (không cần scale) | Bounding-box `left_wheel.STL` = 0.17 x 0.17 x 0.06 m → bán kính 0.085 m, KHỚP CHÍNH XÁC `wheel_diameter=0.17` trong Gazebo plugin của URDF gốc |
| Khoảng cách 2 tâm bánh | 0.3259 m | Đo trực tiếp từ `body_pos` sau khi compile MJCF, khớp `wheel_separation=0.326` trong URDF |
| URDF khai báo `scale="0.001..."` cho `base_footprint.STL` | Đây là LỖI dư thừa trong URDF gốc | Nếu áp dụng thêm scale 0.001, chassis sẽ co còn ~0.7 mm — vô lý; bounding-box đã đúng mét ngay từ đầu |
| Cấu trúc khớp | root free-joint `mobile_base` (= `base_footprint` cũ) → con cứng `base_link` → 2 khớp hinge `left_wheel`/`right_wheel` + 4 sphere caster không khớp | Compile OK, `nbody=10 njnt=3 nu=2` |
| Bánh xe va chạm giả với khung | Có, gây dao động `wheel_qvel` từ -140 đến +115 rad/s dưới lệnh hằng 3 rad/s | `left_wheel`/`right_wheel` là ANH EM của `base_link` (không phải cha–con) nên MuJoCo không tự loại trừ va chạm → cần `<contact><exclude>` |
| RK4 + kv=40 (gain gốc của Bunker) trên quán tính bánh THẬT (0.0023 kg·m²) | Mất ổn định số học (dao động không tắt) | Test trực tiếp: dao động không hội tụ qua 600 bước |
| Nguyên nhân gốc của mất ổn định | Tỉ lệ quán tính bánh/khung quá lệch (khung 58 kg vs bánh 0.0023 kg·m² — nhỏ hơn Bunker ~190 lần) tại dt=0.01s | Test cô lập: dùng `integrator="implicitfast"` (ổn định vô điều kiện với các số hạng giảm chấn tuyến tính) giải quyết được dao động, nhưng vẫn còn trượt bánh nặng (~97%) nếu giữ quán tính thật |
| Trượt bánh khi giữ nguyên quán tính bánh gốc URDF | ~97% (bánh quay nhưng xe gần như đứng yên) | Đo trực tiếp `base_vx` (từ `qvel` của free-joint, KHÔNG suy diễn) sau 30s: chỉ 0.002-0.007 m/s trong khi lăn-không-trượt lý thuyết cho ra 0.24 m/s |
| Khắc phục | Nâng quán tính quay hiệu dụng của bánh lên mức tương đương Bunker gốc (mass=2.0, diag≈0.44/0.72) | Cho steady-state ỔN ĐỊNH, LẶP LẠI ĐƯỢC: `base_vx` khóa đúng 0.1403 m/s suốt từ giây thứ 3 đến giây 27 (không trôi, không nổ số) |
| Dấu quy ước động học vi sai | Đúng chuẩn `ω = (v_phải - v_trái)/track` | Test quay tại chỗ: trái=+2, phải=-2 → yaw quay ÂM (theo chiều kim đồng hồ), đúng như dự đoán từ công thức |

## 4. Root Cause (tại sao Bunker chạy được mà copy y hệt gain lại không)

Bunker dùng bánh **giả lập nặng hơn thực tế** (mass=2 kg, diag inertia
≈0.44 kg·m² cho MỘT bánh xe nhựa nhỏ — vốn dĩ đã là một con số "thổi
phồng" so với vật lý thật, nhưng ổn định về mặt số học). Robot vi sai
của bạn dùng số liệu THẬT từ URDF (mass=1.21 kg, inertia≈0.0023 kg·m²
— nhẹ hơn ~190 lần). Với khung xe nặng 58 kg và timestep vật lý 0.01s,
tỉ lệ quán tính này tạo ra một hệ phương trình ràng buộc "cứng"
(numerically stiff) mà `RK4` (explicit) không giải ổn định được, và
ngay cả `implicitfast` cũng chỉ giải được phần dao động chứ không giải
được vấn đề bánh-quay-không-lăn (đây là vấn đề LỰC MA SÁT KHÔNG ĐỦ THỜI
GIAN TRONG MỘT BƯỚC để truyền động lượng sang khung nặng — bản chất là
vấn đề rời rạc hoá số học, không phải bug logic MJCF).

## 5. Technical Options

| Phương án | Ưu điểm | Nhược điểm |
|---|---|---|
| A. Giữ nguyên inertia thật, giảm `timestep` | Trung thực vật lý nhất | Đã thử tới dt=0.001s vẫn còn ~90% trượt → cần dt cực nhỏ, tốn compute, không thực tế cho RL training (hàng triệu step) |
| B. Nâng `diaginertia`/`mass` bánh lên mức Bunker (ĐANG DÙNG) | Ổn định, lặp lại được, đã kiểm chứng qua 27s mô phỏng liên tục | Thêm "quán tính phản xạ" giả định (motor+hộp số) — hợp lý về kỹ thuật nhưng là THAM SỐ CẦN TINH CHỈNH THÊM, không phải số đo trực tiếp từ URDF |
| C. Actuator dạng `motor` (torque trực tiếp) + PID tự viết trong Python | Kiểm soát được đường cong đáp ứng chi tiết hơn | Phải viết lại toàn bộ `BunkerVelocityController`, tốn công hơn, chưa kiểm chứng trong phiên làm việc này |

## 6. Recommended Implementation Direction

Dùng phương án B làm baseline (đã đóng gói sẵn trong `mobile_body.xml`),
vì đây là cấu hình DUY NHẤT trong toàn bộ các thử nghiệm đạt được
**steady-state ổn định, lặp lại, không NaN, không dao động** — điều kiện
tiên quyết để training SAC/HER hội tụ được. Slip ~40-50% ở steady state
là hệ quả của việc nâng inertia (không phải lỗi), và về bản chất **không
ảnh hưởng đến tính đúng đắn của bài toán RL**: SAC không cần biết quan
hệ `(v,ω)_lệnh → chuyển động thật` là tuyến tính hay có hệ số suy giảm,
nó học trực tiếp từ tương tác. Cái DWA/SAC cần là quan hệ này **ổn định
và lặp lại được giữa các episode** — điều đã được xác nhận.

## 7. Expected Benefits

- Không cần sửa `sensor_utils.py`, `DWA.py`, `feature_extractor.py`,
  `dynamic_obstacles_controller.py`, reward pipeline — vì tên body gốc
  và số tia LiDAR giữ nguyên.
- `BunkerVelocityController` tái sử dụng nguyên vẹn (công thức vi sai
  không phụ thuộc số bánh mỗi bên).

## 8. Risks / Trade-offs

- **Slip ~40-50%** ở steady-state nghĩa là quan hệ giữa `ctrl` (rad/s
  bánh) và vận tốc thật của robot có hệ số tỉ lệ khác Bunker — `DWA.py`
  dùng `max_speed=0.5` (m/s, đơn vị VẬT LÝ THẬT của robot, không phải
  rad/s bánh) nên **không bị ảnh hưởng trực tiếp**, nhưng bạn nên chạy
  lại `dwa_standalone_demo.py` để xem robot mới có đạt được `v=0.5 m/s`
  hay bị giới hạn thấp hơn do slip — nếu thấp hơn nhiều, cần tăng
  `ctrlrange`/`kv` trong `mobile_actuators.xml` để bù.
- Caster mô phỏng bằng sphere ma sát thấp — không mô phỏng góc xoay
  caster thật, chấp nhận được cho bài toán nav 2D nhưng KHÔNG dùng được
  nếu sau này bạn cần mô phỏng động lực học caster chi tiết.

## 9. Validation Experiment (bắt buộc chạy trước khi train)

```bash
pip install mujoco --break-system-packages   # nếu chưa có
python3 scripts/validate_mobile_robot.py --xml assets/worlds/mobile_empty.xml
```

Script in ra 5 bước kiểm tra (compile, đối chiếu Gazebo ground-truth,
đứng yên ổn định, lái thẳng ổn định + % trượt, quay đúng dấu). Nếu bạn
tiếp tục chỉnh `kv`/`damping`/`diaginertia`, LUÔN chạy lại script này
trước khi đụng vào `gym_bunker_env.py`.

Sau đó, kiểm tra bằng mắt:
```bash
python3 assets/worlds/mujoco_viewer.py   # sửa path load sang mobile_empty.xml
```
hoặc chạy `dwa_standalone_demo.py --xml assets/worlds/mobile_empty.xml
--auto-start --render` sau khi patch theo `patches/dwa_standalone_demo.py.patch_instructions.md`.

## 10. Implementation — Checklist các bước cần làm trên repo thật

1. Copy `assets/mobile/` (toàn bộ trong gói này) vào repo, ngang hàng
   với `assets/bunker/` hiện có.
2. Chạy `scripts/convert_world_files.sh` từ root repo để tự động sửa
   toàn bộ 40+ world xml (train/test/val) sang include `mobile_*.xml`
   thay vì `bunker_*.xml`, kèm đổi `integrator="RK4"` →
   `integrator="implicitfast"`. **Luôn `git diff` review trước khi commit.**
3. Áp dụng `patches/gym_bunker_env.py.patch_instructions.md` (3 chỗ sửa).
4. Áp dụng `patches/dwa_standalone_demo.py.patch_instructions.md` nếu
   bạn còn dùng script demo standalone này.
5. Chạy `scripts/validate_mobile_robot.py` để xác nhận model mới OK.
6. Chạy thử 1 episode DWA thuần (`Run_DWA_Only.py` hoặc
   `dwa_standalone_demo.py`) để xem hành vi thực tế trước khi bắt đầu
   train SAC lại từ đầu — **model cũ (`best_model.zip`) sẽ KHÔNG dùng
   lại được** vì `n_lidar` giữ nguyên nhưng động lực học robot đã đổi
   hoàn toàn, chính sách cũ sẽ không còn tối ưu.

## 11. CẬP NHẬT (sau phản hồi thực tế từ bạn dùng)

Hai vấn đề phát sinh khi chạy thử bản đầu, đã sửa trong bản file đính
kèm lần này:

### 11.1 Trục quay bánh xe bị lệch 90° (chỉ ảnh hưởng MESH HIỂN THỊ)

**Nguyên nhân:** bản đầu copy nguyên `euler="1.5708 0 0"` từ quy ước của
Bunker (mesh bánh Bunker được model theo trục Z) sang cho cả body bánh
xe robot vi sai. Nhưng kiểm tra bounding-box `left_wheel.STL` cho thấy
trục "mỏng" (trục quay thật) của mesh này là **trục Y** (bbox
0.1697 x 0.0600 x 0.1700 m), khớp đúng với `<axis xyz="0 1 0"/>` trong
URDF gốc — KHÔNG cần xoay gì thêm. Việc xoay thừa 90° làm mesh hiển thị
sai hướng (dù khớp vật lý/hình trụ va chạm vẫn được set đúng tay nên
robot vẫn lăn đúng — đây là lý do test vật lý pass nhưng nhìn bằng mắt
thấy trục "sai").

**Đã sửa:** bỏ `euler` trên body bánh xe, đặt `joint axis="0 1 0"`
giống hệt URDF gốc; chỉ xoay riêng geom hình trụ (collision proxy) bằng
`euler` cục bộ trên chính geom đó.

### 11.2 Robot "bập bênh" (lắc trước–sau/pitch) ở tốc độ cao

**Nguyên nhân (đã đo trực tiếp, không suy đoán):** để sửa hiện tượng
trượt bánh ở bản đầu, mình đã bơm khống TOÀN BỘ tensor quán tính bánh
xe lên mức của Bunker (`mass=2.0, diaginertia≈0.44/0.72`). Cách này giải
quyết được vấn đề số học, nhưng biến bánh xe thành một "flywheel" nặng —
khi quay nhanh, nó sinh ra mô-men con quay hồi chuyển (gyroscopic torque)
đủ lớn để làm khung xe lắc pitch. Đo được: ở `ctrl=10 rad/s`, pitch dao
động biên độ đỉnh-đỉnh **8.76°** — đúng như hiện tượng "bập bênh" bạn
mô tả.

**Đã sửa:** dùng thuộc tính `armature` trên joint bánh xe thay vì bơm
khống `diaginertia`. `armature` là cơ chế MuJoCo dành RIÊNG để mô phỏng
quán tính phản xạ của motor + hộp số — nó chỉ cộng thêm vào ma trận khối
lượng CỦA DOF ĐÓ (giúp ổn định số học actuator↔khung nặng), KHÔNG cộng
vào tensor quán tính không gian của thân bánh xe, nên KHÔNG gây khớp nối
con quay hồi chuyển vào chuyển động pitch của khung. Giữ nguyên
`mass`/`diaginertia` THẬT từ URDF gốc.

Sau khi dò `armature`/`kv` (script + kết quả trong chat), chốt cấu hình:
`armature="0.15"`, `damping="0.3"` (joint), `kv="150"` (actuator).
Kết quả: pitch peak-to-peak giảm còn **0.50°** (giảm ~17 lần).

| | Trước (bơm diaginertia) | Sau (dùng armature) |
|---|---|---|
| Pitch peak-to-peak @ ctrl=10 rad/s | 8.76° | **0.50°** |
| Trục quay mesh hiển thị | Lệch 90° | Đúng |
| Ổn định số học (không NaN) | Có | Có |

## 12. CẬP NHẬT LẦN 2 (rung/bập bênh ở tốc độ cao + bug LiDAR)

### 12.1 "Rung/bập bênh, bật qua bật lại" — CHỈ xuất hiện khi tăng tốc

**Triệu chứng bạn mô tả:** tốc độ chậm/quay tại chỗ chậm thì êm, tăng
tốc thì rung mạnh như bập bênh. Đây là dấu hiệu kinh điển của
**over-constrained contact chattering**, không phải hiệu ứng con quay
hồi chuyển đã sửa ở mục 11 (2 vấn đề khác nhau, xảy ra đồng thời).

**Đo được (FACT):** robot có 6 điểm tiếp xúc CỨNG đồng thời (2 bánh + 4
caster) — về mặt hình học, 1 mặt phẳng chỉ cần 3 điểm để xác định, nên
đây là hệ dư ràng buộc. Ở tốc độ cao, lực tiếp xúc đo được dao động từ
~80N (tĩnh) lên tới **1042N** (đỉnh), độ lệch chuẩn vận tốc tịnh tiến
`vx_std` tăng theo tốc độ điều khiển — khớp chính xác mô tả "chậm thì
êm, nhanh thì rung" của bạn.

**Đã sửa (2 bước, cả 2 đều áp dụng cho geom caster):**
1. `solref="0.1 1"` — làm mềm contact hơn mặc định (0.02 1) cho cả bánh
   xe và caster.
2. `condim="1" priority="1"` trên caster — chỉ giữ lực pháp tuyến (bỏ
   ràng buộc ma sát tiếp tuyến dư thừa), và ép solver dùng đúng giá trị
   này thay vì bị quy tắc "lấy MAX với floor" ghi đè.

**Kết quả (đo lại, quét toàn bộ dải tốc độ 0.5 → 15 rad/s):**

| | Trước | Sau |
|---|---|---|
| `vx_std` (độ rung vận tốc) | 0.085 (rung nặng) | **0.00000** ở MỌI tốc độ |
| Lực tiếp xúc đỉnh | 1042 N | **~80 N** (ổn định, không dao động) |
| Slip (hiệu ứng phụ) | 40-97% tuỳ cấu hình | **~0.4%** (gần lăn không trượt hoàn hảo) |

### 12.2 Bug LiDAR — tự va chạm với chính robot

**Nguyên nhân (FACT, đo trực tiếp):** hàm raycast trong `sensor_utils.py`
chỉ loại trừ 1 body (`mobile_base`) khỏi kết quả, nhưng hình học thật của
robot (chassis cao ~1.4m, laser, imu, caster) nằm trên body con
`base_link` — KHÔNG được loại trừ. Kiểm tra trong phòng trống: **449/449
tia LiDAR tự va vào chính thân robot** ở khoảng cách ~0.126m.

**Đã sửa:** dùng cơ chế lọc theo `geomgroup` của `mj_ray` (lọc theo
group, không phụ thuộc cây body) thay vì `bodyexclude` (chỉ nhận 1 id).
Cần sửa **CẢ 2 PHÍA**:
- XML (đã áp dụng trong `mobile_body.xml` đính kèm): gán `group="2"`
  cho mọi geom thuộc về robot.
- Python (**bạn cần tự áp dụng** — xem
  `patches/sensor_utils.py.patch_instructions.md`): sửa
  `get_robot_state_and_lidar()` để truyền `geomgroup` loại trừ group 2,
  đồng thời sửa `sensor_height` default từ `0.25` → `0.3488` (đúng độ
  cao lắp LiDAR thật của robot này).

**Kiểm chứng:** phòng trống → 0/449 tia hit (trước: 449/449); có vật cản
thật cách 3m → LiDAR báo đúng góc (~0°) và đúng khoảng cách (2.800m).
Chạy `python3 scripts/test_lidar_fix.py --xml assets/worlds/mobile_empty.xml`
để tự kiểm chứng.

**LƯU Ý QUAN TRỌNG:** đây là lỗi ảnh hưởng trực tiếp đến observation của
SAC (LiDAR sai → toàn bộ 449 giá trị quan sát sai) — nếu bạn đã train
thử ở bước nào đó trước khi áp patch này, kết quả training đó không có
giá trị tham khảo, cần train lại sau khi patch `sensor_utils.py`.


