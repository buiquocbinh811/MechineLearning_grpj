import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Định nghĩa lớp TreeNode để đại diện cho mỗi nút trong cây quyết định
class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids           # Chỉ số của các điểm dữ liệu tại nút này
        self.entropy = entropy   # Entropy tại nút này, sẽ được tính sau
        self.depth = depth       # Độ sâu của nút từ gốc cây
        self.split_attribute = None # Thuộc tính được chọn để phân chia, nếu không phải nút lá
        self.children = children # Danh sách các nút con
        self.order = None        # Thứ tự các giá trị thuộc tính trong các nút con
        self.label = None        # Nhãn của nút nếu nó là nút lá

    # Thiết lập thuộc tính cho các nút không phải nút lá
    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    # Thiết lập nhãn cho nút (cho các nút lá)
    def set_label(self, label):
        self.label = label

# Hàm tính entropy dựa trên phân phối tần suất của các nhãn
def entropy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log2(prob_0))

# Định nghĩa lớp DecisionTreeID3 để xây dựng cây quyết định sử dụng thuật toán ID3
class DecisionTreeID3(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.root = None
        self.max_depth = max_depth  # Độ sâu tối đa của cây
        self.min_samples_split = min_samples_split  # Số mẫu tối thiểu để chia nhánh
        self.Ntrain = 0
        self.min_gain = min_gain    # Lợi ích tối thiểu để chia nhánh
    
    # Huấn luyện cây quyết định trên dữ liệu huấn luyện
    def fit(self, data, target):
        self.Ntrain = data.shape[0]
        self.data = data 
        self.attributes = list(data.columns)
        self.target = target 
        self.labels = target.unique()
        
        # Khởi tạo nút gốc với các chỉ số dữ liệu
        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
        queue = [self.root]
        
        # Xây dựng cây theo chiều rộng
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # Kiểm tra nếu là nút lá
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    # Tính entropy cho một tập hợp các chỉ số dữ liệu
    def _entropy(self, ids):
        if len(ids) == 0: 
            return 0
        freq = np.array(self.target.iloc[ids].value_counts())
        return entropy(freq)

    # Gán nhãn cho nút (dành cho nút lá)
    def _set_label(self, node):
        target_ids = [i for i in node.ids]
        # mode để tính nhãn phổ biến nhất
        node.set_label(self.target.iloc[target_ids].mode()[0])
    
    # Phân chia một nút thành các nút con bằng cách tìm thuộc tính có lợi ích thông tin cao nhất
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        
        # Duyệt qua các thuộc tính để tìm thuộc tính có lợi ích thông tin cao nhất
        for i, att in enumerate(self.attributes):
            values = sub_data[att].unique().tolist()
            if len(values) == 1: 
                continue
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append(sub_ids)
            if min(map(len, splits)) < self.min_samples_split:
                continue
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: 
                continue
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        
        # Thiết lập thuộc tính của nhánh tốt nhất và tạo các nút con
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split, entropy=self._entropy(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    # Dự đoán nhãn cho dữ liệu mới
    def predict(self, new_data):
        npoints = new_data.shape[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]
            node = self.root
            # Duyệt cây cho đến khi đến nút lá
            while node.children: 
                try:
                    node = node.children[node.order.index(x[node.split_attribute])]
                except ValueError:
                    labels[n] = 0
                    break
            else:
                labels[n] = node.label
        return labels

    # Đánh giá độ chính xác của mô hình trên dữ liệu kiểm tra
    # tổng các dự đoán chính xác chia cho tổng số dự đoán
    def evaluate_accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy

    # Tính sai số trung bình tuyệt đối (MAE) giữa dự đoán và nhãn thực tế
    def mean_absolute_error(self, X_test, y_test):
        predictions = self.predict(X_test)
        predictions = np.array([pred if pred is not None else 0 for pred in predictions])
        # lấy trung bình của giá trị tuyệt đối của hiệu giữa dự đoán và nhãn thực tế
        mae = np.mean(np.abs(predictions - y_test))
        return mae


    # Tính R² (hệ số xác định) độ biến nói lên độ nắm bắt dữ liệu
    def r_squared(self, X_test, y_test):
        # Đo lường độ biến thiên của mô hình so với dữ liệu thực tế
        predictions = self.predict(X_test)
        # tính tổng bình phương sai số (SSres) và tổng bình phương tổng quat (SStot)
        ss_res = np.sum((y_test - predictions) ** 2)
        # tổng nhãn thực tế so với trung bình của nhãn thực tế
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
        r2 = max(0, min(1, r2))
        print(f"R²: {r2:.2f}")
        return r2


# Hàm vẽ biểu đồ MAE so với độ sâu của cây
def plot_mae_vs_depth(X_train, y_train, X_test, y_test, max_depth_range):
    maes = []
    for depth in max_depth_range:
        tree = DecisionTreeID3(max_depth=depth, min_samples_split=2)
        tree.fit(X_train, y_train)
        mae = tree.mean_absolute_error(X_test, y_test)
        maes.append(mae)
        print(f"Depth: {depth}, MAE: {mae:.2f}")
    plt.plot(max_depth_range, maes, marker='o')
    plt.xlabel('Tree Depth')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE vs Tree Depth')
    plt.grid()
    plt.show()

# Hàm dự đoán kết quả trận đấu dựa trên các đội nhập vào
def predict_match_result(tree, home_team, away_team, data):
    home_team = home_team.strip().title()
    away_team = away_team.strip().title()

    if home_team not in data['HomeTeam'].values or away_team not in data['AwayTeam'].values:
        print("Một trong hai đội không tồn tại trong dữ liệu.")
        return None

    match_data = data[(data['HomeTeam'] == home_team) & (data['AwayTeam'] == away_team)]
    if match_data.empty:
        print("Trận đấu không có trong dữ liệu.")
        return None

    
    features = match_data[['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']]
    predictions = tree.predict(features)
    if predictions[0] == 1:
        print("Dự đoán: Đội nhà thắng!")
    else:
        print("Dự đoán: Đội nhà không thắng!")
    return predictions[0]

# Chạy chương trình chính để huấn luyện và đánh giá mô hình
if __name__ == "__main__":
    # Đọc dữ liệu
    data = pd.read_csv('D:/Workspace/HocMay/Nhóm 6 - 64KTPM2/final_dataset.csv')
    
    # Chuyển đổi kết quả trận đấu thành nhãn nhị phân (thắng sân nhà = 1, ngược lại = 0)
    data['result'] = data['FTR'].apply(lambda x: 1 if x == 'H' else 0)
    data['HomeTeam'] = data['HomeTeam'].str.title()
    data['AwayTeam'] = data['AwayTeam'].str.title()

    # Chọn các đặc trưng và nhãn
    X = data[['FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts']]
    y = data['result']
    
    # Lựa chọn các đặc trưng mở rộng cho mô hình
    # FTHG: Số bàn thắng của đội nhà.
    # FTAG: Số bàn thắng của đội khách.
    # HTGS: Số bàn thắng trung bình của đội nhà (Home Team Goals Scored).
    # ATGS: Số bàn thắng trung bình của đội khách (Away Team Goals Scored).
    # HTGD: Hiệu số bàn thắng của đội nhà (Home Team Goal Difference).
    # ATGD: Hiệu số bàn thắng của đội khách (Away Team Goal Difference).
    # DiffPts: Chênh lệch điểm số giữa hai đội.
    # DiffFormPts: Chênh lệch điểm số theo phong độ gần đây của hai đội.
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train_size = int(0.7 * len(data))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Khởi tạo và huấn luyện cây quyết định
    tree = DecisionTreeID3(max_depth=3, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    # Đánh giá mô hình
    tree.evaluate_accuracy(X_test, y_test)
    tree.mean_absolute_error(X_test, y_test)
    tree.r_squared(X_test, y_test)
    
    # Nhập tên đội nhà và đội khách để dự đoán
    home_team = input("Nhập tên đội nhà: ")
    away_team = input("Nhập tên đội khách: ")
    
    # Dự đoán kết quả trận đấu
    predict_match_result(tree, home_team, away_team, data)

    # Vẽ biểu đồ MAE so với độ sâu của cây
    max_depth_range = range(1, 11)
    plot_mae_vs_depth(X_train, y_train, X_test, y_test, max_depth_range)

