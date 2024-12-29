import numpy as np
import pandas as pd

# Hàm sigmoid để tính xác suất cho đầu ra nhị phân
def sigmoid(S):
    return 1 / (1 + np.exp(-S))

# Hàm tính xác suất dựa trên trọng số và dữ liệu đầu vào
def prob(w, X):
    return sigmoid(X.dot(w))

# lam là hệ số chặn để tránh overfitting
def loss(w, X, y, lam):
    z = prob(w, X)
    epsilon = 1e-15
    z = np.clip(z, epsilon, 1 - epsilon)  # Giới hạn z để tránh log(0)
    return -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z)) + 0.5 * lam * np.sum(w**2)

# Hàm dự đoán nhãn dựa trên trọng số đã học và ngưỡng 
def predict(w, X, threshold=0.5):
    return (prob(w, X) >= threshold).astype(int)

# Hàm huấn luyện Logistic Regression bằng cách tối ưu hóa trọng số w
def logistic_regression(w_init, X, y, lam=0.001, lr=0.1, nepoches=2000):
    N, d = X.shape
    w = w_init
    loss_hist = [loss(w, X, y, lam)]

    for ep in range(nepoches):
        z = prob(w, X)
       # tối ưu hàm mục tiêu
        gradient = X.T.dot(z - y) / N + lam * w  # Gradient with regularization để tính trọng số w
        w -= lr * gradient  # Cập nhật trọng số
        loss_hist.append(loss(w, X, y, lam))

        if np.linalg.norm(gradient) < 1e-6:  # Điều kiện hội tụ
            break

    return w, loss_hist

# Hàm tính Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Hàm tính R² (Coefficient of Determination)
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Đọc dữ liệu từ file CSV
data = pd.read_csv('D:/Workspace/HocMay/Nhóm 6 - 64KTPM2/final_dataset.csv')

# Chuyển đổi cột kết quả trận đấu thành nhãn (thắng đội nhà = 1, thua hoặc hòa = 0)
data['result'] = data['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Chuyển cột HomeTeam và AwayTeam thành chữ thường
data['HomeTeam'] = data['HomeTeam'].str.lower()
data['AwayTeam'] = data['AwayTeam'].str.lower()


X = data[['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']].values  # Các cột mở rộng
y = data['result'].values

# Chuẩn hóa dữ liệu
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X = (X - mean_X) / std_X  # Chuẩn hóa mỗi cột bằng z-score

# Thêm bias trực tiếp vào X bằng cách thêm một cột 1s ở cuối
X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

# Khởi tạo trọng số ngẫu nhiên
w_init = np.random.randn(X.shape[1])

# Huấn luyện mô hình logistic regression với dữ liệu
lam = 0.001
w, loss_hist = logistic_regression(w_init, X, y, lam, lr=0.05, nepoches=5000)

# In ra trọng số và loss cuối cùng
print('Trọng số cuối cùng của Logistic Regression:', w)
print('Loss cuối cùng:', loss(w, X, y, lam))

# Tính độ chính xác
y_pred = predict(w, X)
accuracy = np.mean(y_pred == y)
print(f'Độ chính xác của mô hình: {accuracy:.2f}')

# Tính MAE và R²
mae = mean_absolute_error(y, y_pred)
r2 = r_squared(y, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R² (Coefficient of Determination): {r2:.2f}')

# Nhập vào tên 2 đội bóng
home_team = input("Nhập tên đội nhà: ").lower()
away_team = input("Nhập tên đội khách: ").lower()

# Tìm thông tin về số bàn thắng và các chỉ số khác của 2 đội trong dữ liệu
home_goals = data[data['HomeTeam'] == home_team]['FTHG'].mean()
away_goals = data[data['AwayTeam'] == away_team]['FTAG'].mean()

home_stats = {
    'HTGS': data[data['HomeTeam'] == home_team]['HTGS'].mean(),
    'HTGD': data[data['HomeTeam'] == home_team]['HTGD'].mean(),
    'DiffPts': data[data['HomeTeam'] == home_team]['DiffPts'].mean(),
    'DiffFormPts': data[data['HomeTeam'] == home_team]['DiffFormPts'].mean(),
}

away_stats = {
    'ATGS': data[data['AwayTeam'] == away_team]['ATGS'].mean(),
    'ATGD': data[data['AwayTeam'] == away_team]['ATGD'].mean(),
}

# Kiểm tra xem có dữ liệu cho các đội bóng không
if np.isnan(home_goals) or np.isnan(away_goals):
    print("Không tìm thấy dữ liệu cho một trong hai đội bóng.")
else:
    # Tạo dữ liệu đầu vào mới với các cột mở rộng
    input_data = pd.DataFrame([[home_goals, away_goals, home_stats['HTGS'], away_stats['ATGS'],
                                home_stats['HTGD'], away_stats['ATGD'],
                                home_stats['DiffPts'], home_stats['DiffFormPts']]], 
                              columns=['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'])
    
    # Chuẩn hóa dữ liệu đầu vào giống như tập huấn luyện
    input_data_scaled = (input_data.values - mean_X) / std_X
    input_data_scaled = np.concatenate((input_data_scaled, np.ones((input_data_scaled.shape[0], 1))), axis=1)  # Thêm bias

    # Dự đoán kết quả trận đấu
    prediction = predict(w, input_data_scaled)

    if prediction == 1:
        print(f"Đội {home_team.capitalize()} thắng.")
    else:
        print(f"Đội {home_team.capitalize()} không thắng.")

# Vẽ đồ thị mất mát
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(loss_hist, label='Loss', color='blue')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
