# Phân loại chữ số viết tay — NV2

Quy trình hiện tại gồm **Logistic Regression, KNN, SVM linear/RBF và Random Forest**.

- [Hướng dẫn NV2 và bàn giao cho nhóm](README_NV2.md)
- [Khai báo mô hình và siêu tham số](src/nv2_models.py)
- [Huấn luyện, GridSearchCV và đánh giá](src/nv2.py)
- [Báo cáo kết quả](outputs/nv2/report.md)
- [Phân tích ảnh dự đoán sai](outputs/nv2/error_analysis.md)
- [PowerPoint thực nghiệm](outputs/nv2/slides/NV2_thuc_nghiem_v2.pptx)

## Cấu trúc

```text
src/
  nv2.py
  nv2_models.py
tests/
  test_nv2.py
data/
  digit-recognizer/
  IAM/
  aug_word_data4.csv/
outputs/
  nv2/                  # Mô hình, bộ chia chung, chỉ số và slide
```

Dữ liệu huấn luyện NV2: `data/digit-recognizer/train.csv`.
Các dataset và notebook OCR khác được giữ nguyên để nhóm tham khảo.

## Chạy

```powershell
# Dùng môi trường .venv đã có; nếu chưa có, tạo và cài requirements.txt
.\.venv\Scripts\python.exe -m src.nv2 run --output outputs/nv2_new

# Xem lại báo cáo từ kết quả đã lưu, không huấn luyện hoặc predict test lại
.\.venv\Scripts\python.exe -m src.nv2 report

# Kiểm tra
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Không ghi đè hoặc tuning lại trên run đã mở tập test. Xem README_NV2.md để chạy
riêng từng bước và bổ sung kết quả CNN từ NV3.
