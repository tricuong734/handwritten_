# NV2 — Nhận xét ảnh dự đoán sai

Đối chiếu 15 ảnh sai đầu tiên của từng mô hình với nhãn gốc. `row_id` là chỉ số
dòng từ 0 trong CSV nguồn, không tính header. Số đếm bên dưới tính trên toàn bộ
6.300 ảnh test; nhận xét hình dáng chỉ dựa trên ảnh mẫu đã xem. Đây là các giả
thuyết giải thích lỗi, không phải kết luận nhân quả hay lý do để thay nhãn gốc.

| Mô hình | Số ảnh sai | Cặp nhầm nhiều nhất |
|---|---:|---|
| Logistic Regression | 540 | 9→7: 27 ảnh; 5→3: 26 ảnh |
| KNN | 210 | 4→9: 13 ảnh; 8→3: 12 ảnh |
| SVM linear | 411 | 5→3: 27 ảnh; 9→4: 26 ảnh |
| SVM RBF | 147 | 9→7, 9→4, 4→9: mỗi cặp 8 ảnh |
| Random Forest | 246 | 4→9: 13 ảnh; 9→7, 9→4, 8→9, 3→2: mỗi cặp 10 ảnh |

## Các tình huống quan sát được

- **Vòng trên nhỏ, thân dọc nổi bật:** ảnh `34007` có nhãn 9, nhưng cả năm mô hình
  đoán 1. Vòng trên hẹp có thể chưa đủ nổi bật so với nét dọc dài. Đây là ví dụ
  cần bàn giao cho C kiểm tra CNN có cải thiện hay không.
- **Hình dáng của 3 gần 2:** ảnh `36227` có nhãn 3, phần trên tạo một nét chéo dài;
  LR, KNN, SVM linear và SVM RBF đều đoán 2. Sự khác nhau ở vị trí nối hai cung
  và nét cuối là một giả thuyết cho việc nhầm.
- **Nét kéo dài của 9 gần 7:** ảnh `9471` có nhãn 9 nhưng LR, KNN, SVM linear và
  SVM RBF đoán 7; nét đuôi dài và chéo, vòng trên dẹt. Điều này phù hợp với việc
  9→7 xuất hiện trong nhóm nhầm nhiều của các mô hình.
- **Kiểu viết 4 gần 9:** ảnh `3791` có nhãn 4; KNN và SVM RBF đoán 9, Random Forest
  đoán 2. Ảnh `11945` cũng là 4 bị SVM RBF đoán 9. Kiểu viết, vị trí nối nét và
  phần vòng ở phía trên có thể ảnh hưởng đến dự đoán.
- **Nét 5 dày, gần hai cung của 3:** ảnh `3672` có nhãn 5; KNN, SVM linear,
  SVM RBF và Random Forest đoán 3. Phần đầu đậm và phần cong phía dưới khiến
  chi tiết phân biệt 5/3 khó nhận rõ ở độ phân giải 28×28.
- **Ảnh có hình dáng bất thường:** ảnh `23231` mang nhãn 0 nhưng có vòng nhọn và
  nét kéo dài; KNN/Random Forest đoán 5, SVM RBF đoán 3. Có thể đánh dấu để kiểm
  tra chất lượng dữ liệu, nhưng không sửa nhãn chỉ vì mô hình đoán khác.

## Nhận xét theo mô hình

**Logistic Regression:** sai nhiều nhất (540 ảnh); một số mẫu 6→7, 3→8, 9→3
cho thấy mô hình chưa phân biệt tốt các kiểu viết biến dạng. Cần dùng số liệu
và ảnh cụ thể trong báo cáo, không suy ra mọi lỗi đều do cùng một nguyên nhân.

**KNN:** ít lỗi hơn LR và SVM linear; vẫn nhầm các ảnh có nét dính hoặc khác
kiểu viết phổ biến như 4/9, 8/3. Thời gian fit rất ngắn nhưng phải lưu các ảnh train,
nên cần xem cả thời gian dự đoán và dung lượng file khi chọn cho ứng dụng.

**SVM linear:** giảm số lỗi so với LR nhưng còn 411 ảnh sai. Các cặp 5/3 và 9/4
vẫn chiếm nhiều lỗi. Trên cùng dữ liệu và cách chia, SVM RBF đạt kết quả tốt hơn.

**SVM RBF:** ít lỗi nhất (147 ảnh), đồng thời được chọn theo validation trước
khi mở test. Những ảnh sai vẫn có các trường hợp hình dáng dễ lẫn; không đạt
độ chính xác tuyệt đối. Đây là lựa chọn hiện tại cho chất lượng dự đoán trong
nhóm mô hình cổ điển, chưa phải kết luận so với CNN.

**Random Forest:** 246 ảnh sai, accuracy thấp hơn KNN nhưng dự đoán cả test nhanh
hơn trong lần đo này. Cần cân nhắc tốc độ và kích thước file thay vì chỉ accuracy.

## Hướng kiểm chứng sau này

Cùng A kiểm tra ảnh lệch tâm, độ nghiêng, độ dày nét và chất lượng nhãn; cùng C
so sánh những `row_id` khó trên CNN. Nếu thử căn giữa, deskew hoặc đặc trưng khác,
chỉ lựa chọn bằng train/validation. Không lặp điều chỉnh trên tập test đã mở.

Xem ảnh và danh sách đầy đủ trong `*/misclassified.png` và `*/misclassified.csv`.
