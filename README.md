---
title: Leaf Vision Lab - Plant Leaf Classification
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.41.0
app_file: streamlit_app.py
pinned: false
---

# PlantVillage Dataset — Leaf Vision Lab Demo

Tài liệu tóm tắt nguồn dữ liệu, tổng quan và các lớp (classes) được sử dụng trong bài toán phân loại loài cây từ ảnh lá (Chương 1 — Tổng quan Thị giác máy tính & Xử lý ảnh số) theo đặc tả `DIS_demo-v1.md`.

---

## 1. Nguồn dữ liệu (Dataset Source)
* **Kho lưu trữ chính thức:** [PlantVillage-Dataset (GitHub)](https://github.com/spMohanty/PlantVillage-Dataset.git)
* **Tác giả:** Sharada P. Mohanty, David P. Hughes, Marcel Salathé
* **Giấy phép:** Creative Commons Zero (CC0)
* **Phương thức tải vào dự án:** Git Sparse-Checkout chỉ tải 2 thư mục:
  * `raw/color`: Chứa ảnh màu nguyên bản (RGB) trên phông nền đơn sắc có kiểm soát.
  * `raw/segmented`: Chứa ảnh đã tách nền đen, dùng làm **Ground Truth** tính chỉ số **IoU / Dice** cho Tab 5 (Segmentation & Morphology).
* **Lưu ý chất lượng:**
  * Đã loại bỏ hoàn toàn `raw/grayscale` để tiết kiệm dung lượng.
  * **Tuyệt đối không dùng bản "New Plant Diseases Dataset (Augmented)"** nhằm ngăn ngừa rò rỉ dữ liệu (Data Leakage) giữa các tập train, validation và test.

---

## 2. Tổng quan bài toán & Tiêu chí chọn lớp
* **Bài toán:** Phân loại **loài cây** (Plant Species Classification) từ ảnh **lá khỏe (Healthy)** nhằm đảm bảo tính đồng nhất về hình thái học của lá, phục vụ so sánh xuyên suốt từ Xử lý ảnh cổ điển đến Deep Learning.
* **Tiêu chuẩn số lượng:** Mỗi loài phải có tối thiểu $\ge 300$ ảnh lá khỏe.

---

## 3. Thống kê thực tế các lớp được sử dụng trong Demo

Kết quả quét và đếm trực tiếp trên dữ liệu tải về tại `demo/dataset/raw/color`:

| STT | Tên lớp trong dataset | Loài cây (Species) | Số ảnh màu (`raw/color`) | Số ảnh tách nền (`raw/segmented`) | Trạng thái kiểm duyệt ($\ge 300$) |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | `Apple___healthy` | Táo (Apple) | 1.645 | 1.645 | Đạt chuẩn |
| 2 | `Blueberry___healthy` | Việt quất (Blueberry) | 1.502 | 1.502 | Đạt chuẩn |
| 3 | `Corn_(maize)___healthy` | Bắp / Ngô (Corn) | 1.162 | 1.162 | Đạt chuẩn |
| 4 | `Grape___healthy` | Nho (Grape) | 423 | 423 | Đạt chuẩn |
| 5 | `Pepper,_bell___healthy` | Ớt chuông (Pepper bell) | 1.478 | 1.478 | Đạt chuẩn |
| 6 | `Tomato___healthy` | Cà chua (Tomato) | 1.591 | 1.591 | Đạt chuẩn |
| **Tổng** | **6 lớp chính thức** | | **7.801** | **7.801** | **Đầy đủ dữ liệu** |

> **Kết luận kiểm tra Phase 1:** Tất cả 6 loài dự kiến ban đầu đều có $\ge 300$ ảnh (thấp nhất là Grape với 423 ảnh). Do đó, **giữ nguyên 6 loài ban đầu**, không cần kích hoạt phương án thay thế dự phòng (Cherry: 854 ảnh, Strawberry: 456 ảnh, Raspberry: 371 ảnh).

---

## 4. Kế hoạch trích xuất tập con (Subset) theo `DIS_demo-v1.md`
* **Quy mô lấy mẫu:** 350 ảnh/loài (ngẫu nhiên với `seed=42`).
* **Tổng số ảnh tập con:** $6 \times 350 = 2.100$ ảnh.
* **Phân chia Stratified Splits (70% / 15% / 15%):**
  * **Train:** 245 ảnh/loài (Tổng: 1.470 ảnh)
  * **Validation:** 52 ảnh/loài (Tổng: 312 ảnh)
  * **Test:** 53 ảnh/loài (Tổng: 318 ảnh) — cố định dùng chung cho cả Tab 7 (ML) và Tab 8 (DL).
