---
sdk: streamlit
sdk_version: 1.41.0
app_file: streamlit_app.py
pinned: false
---
## 1. Nguồn dữ liệu (Dataset Source)
* **Kho lưu trữ chính thức:** [PlantVillage-Dataset (GitHub)](https://github.com/spMohanty/PlantVillage-Dataset.git)
* **Tác giả:** Sharada P. Mohanty, David P. Hughes, Marcel Salathé
* **Giấy phép:** Creative Commons Zero (CC0)
* **Phương thức tải vào dự án:** Git Sparse-Checkout chỉ tải 2 thư mục:
  * `raw/color`: Chứa ảnh màu nguyên bản (RGB) trên phông nền đơn sắc có kiểm soát.
  * `raw/segmented`: Chứa ảnh đã tách nền đen, dùng làm **Ground Truth** tính chỉ số **IoU / Dice** cho Tab 5 (Segmentation & Morphology).
