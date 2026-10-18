# NV2 — Huấn luyện, đánh giá và PowerPoint

Phạm vi theo bản phân công chính: Logistic Regression, KNN, SVM linear/RBF,
Random Forest; GridSearchCV; đo thời gian; đánh giá từng lớp; phân tích ảnh sai;
lưu model; tổng hợp slide thực nghiệm. Không viết phần lý thuyết.

## Mã nguồn

- `src/nv2_models.py`: khai báo từng mô hình và lưới siêu tham số. Logistic Regression nằm ngay đầu hàm `model_specs()`.
- `src/nv2.py`: chuẩn bị dữ liệu, train, đánh giá test và tạo báo cáo.
- `tests/test_nv2.py`: kiểm tra cách chia dữ liệu, ghép dự đoán CNN và ngăn train lại sau khi mở test.

Đây là quy trình huấn luyện và đánh giá hiện tại của dự án.

## Kết quả hiện tại

Xem `outputs/nv2/report.md`, `outputs/nv2/test_comparison.csv` và
`outputs/nv2/error_analysis.md`. Mỗi thư mục mô hình có:

- `model.joblib`: mô hình đã fit, nhận float32 shape `(n, 784)`, pixel trong `[0,1]`.
- `training.json`: tham số chọn, điểm CV, thời gian tìm tham số và fit cuối, kích thước file, cảnh báo nếu có.
- `cv_results.csv`: kết quả từng cấu hình và từng fold do GridSearchCV xuất.
- `validation_predictions.csv`, `test_predictions.csv`: nhãn thật và dự đoán, có `row_id` để truy lại ảnh gốc.
- `test_metrics.json`: chỉ số tổng thể, thời gian dự đoán test và ms/ảnh.
- `test_classification_report.csv`: precision/recall/F1/support theo lớp; thêm OvR accuracy.
- `test_confusion_matrix.csv/png`, `misclassified.csv/png`, `error_summary.json`.

`row_id` đếm dòng dữ liệu từ 0, không tính header. Precision/Recall/F1 tổng hợp dùng macro average.
Accuracy tổng thể là tỷ lệ đúng trên toàn bộ ảnh. `ovr_accuracy` tính mỗi chữ số so với phần còn lại,
bao gồm true negatives nên không đồng nghĩa tỷ lệ đúng riêng của lớp đó; dùng recall để xem tỷ lệ đúng
trong các ảnh thật thuộc lớp đó.

## Dữ liệu và thiết kế thí nghiệm

Nguồn: `data/digit-recognizer/train.csv`, 42.000 ảnh gắn nhãn.
Chia phân tầng cố định seed 42: **29.400 train / 6.300 validation / 6.300 test**.
`data/digit-recognizer/test.csv` không có nhãn nên không dùng để tính chỉ số.

Chuẩn hóa pixel /255, không augmentation cho lần so sánh mô hình cổ điển này.
GridSearchCV 3 fold trên cùng 6.000 ảnh train lấy mẫu phân tầng; fit cuối trên toàn bộ 29.400 ảnh train.
Chọn mô hình theo validation trước khi mở test. Không fit lại bằng train+validation trong lần chạy này.
Lưới tìm kiếm:

| Mô hình | Siêu tham số |
|---|---|
| Logistic Regression | C = 0.1, 1, 10 |
| KNN | k = 3, 5, 7; weights = uniform, distance |
| SVM linear | C = 0.1, 1 |
| SVM RBF | C = 1, 5; gamma = scale, 0.01 |
| Random Forest | 150 cây; max_depth = None, 25; max_features = sqrt, 0.15 |

CV dùng để chọn tham số, không phải nested CV. Tập test được giữ riêng trong quy trình mới này,
nhưng nguồn dữ liệu đã được dùng trong các thí nghiệm trước của dự án; không coi đây là dữ liệu bên ngoài hoàn toàn mới.
Thời gian đo phụ thuộc máy, mặc định giới hạn 2 luồng BLAS; estimator và GridSearchCV chạy n_jobs=1.
Thời gian tìm kiếm được báo cáo riêng khỏi thời gian fit cuối. KNN lưu mẫu khi fit nên thời gian fit rất ngắn;
chi phí chính nằm ở dự đoán. Kích thước model là dung lượng joblib nén mức 3.

## Chạy lại

Trong thư mục dự án, dùng PowerShell:

```powershell
# Cài thư viện nếu chưa có môi trường
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Tạo lần thí nghiệm mới, không ghi đè outputs/nv2 hiện có
.\.venv\Scripts\python.exe -m src.nv2 run --output outputs/nv2_new

# Hoặc chạy từng giai đoạn; cùng --output cho cả ba lệnh
.\.venv\Scripts\python.exe -m src.nv2 prepare --output outputs/nv2_staged
.\.venv\Scripts\python.exe -m src.nv2 train --output outputs/nv2_staged
.\.venv\Scripts\python.exe -m src.nv2 evaluate --output outputs/nv2_staged

# Chỉ tái tạo báo cáo từ kết quả đã lưu, không predict test lại
.\.venv\Scripts\python.exe -m src.nv2 report

# Kiểm thử
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các tùy chọn chuẩn bị: `--seed`, `--folds`, `--tune-samples` (0 = toàn bộ train),
`--threads`, `--max-samples` (chỉ dành cho smoke test), `--data`, `--output`.
Thay đổi các lựa chọn này bằng một lần `prepare` mới. `train` đọc cấu hình đã lưu.
Nếu train bị gián đoạn, chạy lại `train` trên cùng thư mục để tiếp tục các mô hình chưa xong.

Sau khi mở test, `test_frozen.json` khóa hash của model, dữ liệu chia và lựa chọn validation.
Không được train/tune lại trên cùng run. Chạy `evaluate` lại chỉ dùng kết quả test đã lưu cho mô hình đã hoàn tất.
Không dùng test để thử lặp tham số rồi chọn điểm cao nhất.

## Bàn giao cho A và C

Dùng `outputs/nv2/shared_data.npz`: `X_train`, `y_train`, `X_validation`, `y_validation`,
`X_test`, `y_test` và các `*_row_ids`. A có thể kiểm tra/chấp nhận cách chia này;
C phải train CNN trên cùng train/validation để so sánh công bằng. `split_indices.npz`
lưu vị trí gốc, `config.json` lưu hash CSV và phiên bản thư viện.

```python
import numpy as np
data = np.load('outputs/nv2/shared_data.npz')
X_train = data['X_train'].reshape(-1, 28, 28, 1)
y_train = data['y_train']
X_val = data['X_validation'].reshape(-1, 28, 28, 1)
y_val = data['y_validation']
```

**Chưa có model/kết quả CNN của C trong thư mục, nên chưa thể hoàn tất đánh giá toàn nhóm.**
Sau khi C chốt CNN bằng train/validation, C chạy predict test đúng một lần, xuất CSV hai cột
`row_id,y_pred` cho đúng 6.300 ảnh test. Không gửi số đo từ một tập test khác.
Metadata JSON do C cung cấp gồm:

- `fit_seconds`: thời gian huấn luyện CNN.
- `predict_seconds`: thời gian dự đoán cả tập test (không gồm tải model).
- `model_bytes`: kích thước file CNN đã lưu.
- `shared_data_sha256`: SHA-256 của chính `shared_data.npz` dùng để train/đánh giá.
- Có thể thêm cấu hình, framework, thiết bị CPU/GPU để giải thích chênh lệch thời gian.

```powershell
.\.venv\Scripts\python.exe -m src.nv2 evaluate-cnn --predictions duong_dan/cnn_predictions.csv --cnn-metadata duong_dan/cnn_metadata.json
```

Lệnh kiểm tra đúng row_id, căn chỉnh thứ tự và xuất cùng bộ chỉ số/ảnh lỗi, rồi cập nhật bảng so sánh.
Chỉ tiếp nhận một lần kết quả CNN trong run này. Model CNN và mã train vẫn do NV3 bàn giao cho A.

## PowerPoint

Slide thực nghiệm đặt trong `outputs/nv2/slides/`. Nội dung lấy từ dữ liệu và kết quả thật,
không chứa lý thuyết. Phần CNN và demo toàn nhóm còn phụ thuộc bàn giao từ A/C.
