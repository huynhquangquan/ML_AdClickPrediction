# Ad Click Prediction ML Project

## Mô tả dự án
Dự án này sử dụng Machine Learning để dự đoán khả năng người dùng nhấp vào quảng cáo dựa trên dữ liệu mạng xã hội. Dự án bao gồm việc khám phá dữ liệu, xây dựng mô hình, và triển khai API cùng giao diện web để dự đoán.

## Tính năng
- Khám phá và phân tích dữ liệu từ tập Social_Network_Ads.csv
- Xây dựng và tối ưu hóa mô hình Random Forest cho dự đoán
- API FastAPI để phục vụ dự đoán
- Giao diện web đơn giản để nhập dữ liệu và nhận kết quả

## Cài đặt
1. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```
2. Đảm bảo có file dữ liệu `Social_Network_Ads.csv` trong thư mục gốc (hoặc tải từ Kaggle nếu cần).

## Sử dụng
1. Mở và chạy notebook `Ad_Click_Prediction_ML.ipynb` để khám phá dữ liệu, huấn luyện mô hình và lưu model/scaler.
2. Khởi động API FastAPI:
   ```
   python "FastAPI (Adclick).py"
   ```
3. Mở file `AdClickPredictionWebsite.html` trong trình duyệt để nhập dữ liệu và nhận kết quả dự đoán.

## Cấu trúc dự án
- `Ad_Click_Prediction_ML.ipynb`: Notebook Jupyter cho phân tích dữ liệu và huấn luyện mô hình
- `FastAPI (Adclick).py`: Script FastAPI cho API dự đoán
- `AdClickPredictionWebsite.html`: Giao diện web
- `random_forest_optimized_adclick.pkl`: Mô hình Random Forest đã huấn luyện
- `Scaler_adclick.pkl`: Scaler cho chuẩn hóa dữ liệu
- `LICENSE.md`: Giấy phép MIT

## Giấy phép
MIT License - Xem file LICENSE.md để biết thêm chi tiết.