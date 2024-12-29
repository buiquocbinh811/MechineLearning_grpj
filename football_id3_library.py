import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('D:/Workspace/HocMay/Nhóm 6 - 64KTPM2/final_dataset.csv')

# Chuyển đổi cột kết quả trận đấu thành nhãn (thắng đội nhà = 1, thua hoặc hòa = 0)
data['result'] = data['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Chuyển cột HomeTeam và AwayTeam thành chữ thường để tránh phân biệt hoa thường
data['HomeTeam'] = data['HomeTeam'].str.lower()
data['AwayTeam'] = data['AwayTeam'].str.lower()

# Lựa chọn các đặc trưng quan trọng cho mô hình
X = data[['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']]
y = data['result']  # Cột kết quả (thắng hoặc không thắng) là biến mục tiêu

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ramdom_state = 42 để cố định việc chia dữ liệu ngẫu nhiên, giúp kết quả của mô hình ổn định

# Tạo mô hình ID3 Decision Tree với tiêu chí entropy để phân chia nút
model = DecisionTreeClassifier(criterion="entropy")

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập dữ liệu kiểm tra
y_pred = model.predict(X_test)

# Tính các độ đo hiệu suất của mô hình
accuracy = accuracy_score(y_test, y_pred)  # Độ chính xác
mae = mean_absolute_error(y_test, y_pred)  # Sai số tuyệt đối trung bình
r2 = r2_score(y_test, y_pred)  # Hệ số xác định R²

# In ra kết quả hiệu suất
print(f'Accuracy: {accuracy:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R² (Coefficient of Determination): {r2:.2f}')

# Nhập vào tên 2 đội bóng để dự đoán kết quả
home_team = input("Nhập tên đội nhà: ").lower()  # Chuyển tên đội về dạng chữ thường
away_team = input("Nhập tên đội khách: ").lower()

# Tìm thông tin về số bàn thắng và các đặc trưng khác của 2 đội trong dữ liệu
home_goals = data[data['HomeTeam'] == home_team]['FTHG'].mean()
away_goals = data[data['AwayTeam'] == away_team]['FTAG'].mean()
home_team_HTGS = data[data['HomeTeam'] == home_team]['HTGS'].mean()
away_team_ATGS = data[data['AwayTeam'] == away_team]['ATGS'].mean()
home_team_HTGD = data[data['HomeTeam'] == home_team]['HTGD'].mean()
away_team_ATGD = data[data['AwayTeam'] == away_team]['ATGD'].mean()
diff_pts = data[(data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)]['DiffPts'].mean()
diff_form_pts = data[(data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)]['DiffFormPts'].mean()

# Kiểm tra xem có dữ liệu cho các đội bóng không (nếu có giá trị NaN thì thiếu dữ liệu)
if np.isnan(home_goals) or np.isnan(away_goals) or np.isnan(home_team_HTGS) or np.isnan(away_team_ATGS) or np.isnan(home_team_HTGD) or np.isnan(away_team_ATGD) or np.isnan(diff_pts) or np.isnan(diff_form_pts):
    print("Không tìm thấy đủ dữ liệu cho một trong hai đội bóng.")
else:
    # Tạo DataFrame cho dữ liệu đầu vào với tên cột phù hợp cho mô hình
    input_data = pd.DataFrame([[home_goals, away_goals, home_team_HTGS, away_team_ATGS, home_team_HTGD, away_team_ATGD, diff_pts, diff_form_pts]], 
                              columns=['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'])
    
    # Dự đoán kết quả trận đấu giữa đội nhà và đội khách
    prediction = model.predict(input_data)
    
    # Hiển thị kết quả dự đoán
    if prediction == 1:
        print(f"Đội {home_team.capitalize()} thắng")
    else:
        print(f"Đội {away_team.capitalize()} không thắng.")
