#!/usr/bin/env bash
# convert_world_files.sh
# ---------------------------------------------------------------
# Chạy 1 lần từ ROOT của repo (rl_nav/) để đổi TẤT CẢ world xml đang
# include bunker_*.xml sang dùng mobile_*.xml.
#
# An toàn: chỉ sed các dòng <include file="...bunker/..."/>, không đụng
# obstacle/goal-marker nào khác. Luôn review bằng `git diff` trước khi
# commit; có thể chạy lại `git checkout -- assets/worlds/` để hoàn tác.
#
# Cách dùng:
#   chmod +x scripts/convert_world_files.sh
#   ./scripts/convert_world_files.sh
# ---------------------------------------------------------------
set -euo pipefail

WORLDS_DIR="assets/worlds"

if [ ! -d "$WORLDS_DIR" ]; then
    echo "Không tìm thấy $WORLDS_DIR -- hãy chạy script này từ thư mục gốc repo."
    exit 1
fi

echo "Đang quét các world xml trong $WORLDS_DIR ..."
COUNT=0
for f in $(find "$WORLDS_DIR" -name "*.xml"); do
    if grep -q "bunker/bunker" "$f" 2>/dev/null; then
        # meshdir: ../../bunker -> ../../mobile  (test/train/val subfolders)
        #          ../bunker    -> ../mobile      (world root, vd empty.xml)
        sed -i \
            -e 's#\.\./\.\./bunker/bunker_assests\.xml#../../mobile/mobile_assets.xml#g' \
            -e 's#\.\./\.\./bunker/bunker_body\.xml#../../mobile/mobile_body.xml#g' \
            -e 's#\.\./\.\./bunker/bunker_actuators\.xml#../../mobile/mobile_actuators.xml#g' \
            -e 's#\.\./bunker/bunker_assests\.xml#../mobile/mobile_assets.xml#g' \
            -e 's#\.\./bunker/bunker_body\.xml#../mobile/mobile_body.xml#g' \
            -e 's#\.\./bunker/bunker_actuators\.xml#../mobile/mobile_actuators.xml#g' \
            -e 's#target="mobile_base"#target="mobile_base"#g' \
            "$f"
        # add the contact-exclude include + switch integrator (only if not already patched)
        if ! grep -q "mobile_contacts.xml" "$f"; then
            # chèn include contact ngay trước </asset> đóng, hoặc sau dòng </asset>
            sed -i "/<\/asset>/a\\
\\
  <include file=\"../mobile/mobile_contacts.xml\"/>" "$f" 2>/dev/null || true
        fi
        sed -i 's#integrator="RK4"#integrator="implicitfast"#g' "$f"
        # meshdir trong <compiler> cũng phải trỏ sang mobile
        sed -i 's#meshdir="\.\./\.\./bunker"#meshdir="../../mobile"#g' "$f"
        sed -i 's#meshdir="\.\./bunker"#meshdir="../mobile"#g' "$f"

        COUNT=$((COUNT+1))
        echo "  patched: $f"
    fi
done

echo ""
echo "Đã patch $COUNT file. Kiểm tra lại bằng:"
echo "  git diff -- $WORLDS_DIR | less"
echo ""
echo "LƯU Ý: file cần thư mục ../../mobile/ hoặc ../mobile/ tồn tại đúng vị trí"
echo "tương ứng với độ sâu của mỗi world xml (train/, test/, val/, hoặc root)."
