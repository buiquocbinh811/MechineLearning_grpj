import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('D:/Workspace/HocMay/Nhóm 6 - 64KTPM2/final_dataset.csv')

# Chuyển đổi cột kết quả trận đấu thành nhãn (thắng đội nhà = 1, thua hoặc hòa = 0)
data['result'] = data['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Chuyển cột HomeTeam và AwayTeam thành chữ thường để tránh phân biệt hoa thường
data['HomeTeam'] = data['HomeTeam'].str.lower()
data['AwayTeam'] = data['AwayTeam'].str.lower()

# Lựa chọn các đặc trưng quan trọng cho mô hình
X = data[['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']].values  # Các đặc trưng đầu vào
y = data['result'].values  # Nhãn mục tiêu

# Chuẩn hóa dữ liệu để giúp mô hình hội tụ tốt hơn giúp mô hình có độ chính xác cao hơn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression(max_iter=1000)

# Khởi tạo StratifiedKFold để chia dữ liệu thành 5 phần, đảm bảo tỷ lệ nhãn ở mỗi phần
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Biến để lưu trữ kết quả trong quá trình cross-validation đánh giá mô hình
accuracies = []
mae_scores = []
r2_scores = []

# Tiến hành cross-validation với 5 lần lặp để đánh giá mô hình
for train_index, test_index in kf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(X_train, y_train)

    # Dự đoán trên tập dữ liệu kiểm tra
    y_pred = model.predict(X_test)

    # Tính độ chính xác và độ sai lệch trung bình và độ biến thiên R² 
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Lưu trữ kết quả của mỗi lần lặp
    accuracies.append(accuracy)
    mae_scores.append(mae)
    r2_scores.append(r2)

# In ra các kết quả sau khi hoàn thành cross-validation
print(f'Độ chính xác trung bình: {np.mean(accuracies):.2f}')
print(f'Mean Absolute Error (MAE): {np.mean(mae_scores):.2f}')
print(f'R² (Coefficient of Determination): {np.mean(r2_scores):.2f}')

# Nhập vào tên 2 đội bóng để dự đoán kết quả
home_team = input("Nhập tên đội nhà: ").lower()
away_team = input("Nhập tên đội khách: ").lower()

# Tìm thông tin về số bàn thắng và các chỉ số khác của 2 đội trong dữ liệu
home_goals = data[data['HomeTeam'] == home_team]['FTHG'].mean()
away_goals = data[data['AwayTeam'] == away_team]['FTAG'].mean()

# Kiểm tra xem có dữ liệu cho các đội bóng không (nếu có giá trị NaN nghĩa là thiếu dữ liệu)
if np.isnan(home_goals) or np.isnan(away_goals):
    print("Không tìm thấy dữ liệu cho một trong hai đội bóng.")
else:
    # Tạo dữ liệu đầu vào mới với số bàn thắng đã chuẩn hóa
    input_data = np.array([[home_goals, away_goals, 0, 0, 0, 0, 0, 0]])  # Các chỉ số HTGS, ATGS, HTGD, ATGD, DiffPts, DiffFormPts bằng 0
    input_data_scaled = scaler.transform(input_data)  # Chuẩn hóa dữ liệu đầu vào

    # Dự đoán kết quả trận đấu giữa đội nhà và đội khách
    prediction = model.predict(input_data_scaled)

    # Hiển thị kết quả dự đoán
    if prediction == 1:
        print(f"Đội {home_team.capitalize()} thắng.")
    else:
        print(f"Đội {home_team.capitalize()} không thắng")

# Import thư viện matplotlib để trực quan hóa dữ liệu
import matplotlib.pyplot as plt

# Khởi tạo biểu đồ với kích thước 12x6
plt.figure(figsize=(12, 6))
