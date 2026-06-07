import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================
# 1. ĐỌC DỮ LIỆU GỐC TỪ GIAI ĐOẠN 3 
# ==========================================
input_path = '../dataset/HanoiHousing_Stage3_Cleaned.csv'
df = pd.read_csv(input_path)

print("=" * 60)
print(f"ĐỌC DỮ LIỆU TỪ STAGE 3: {df.shape[0]} dòng | {df.shape[1]} cột")
print("=" * 60 + "\n")

# ==========================================
# 2. [KỸ THUẬT 1] CẮT BỎ NGOẠI LAI TÀN NHẪN (AGGRESSIVE TRIMMING)
# ==========================================
print("[1/4] Đang cắt bỏ ngoại lai tàn nhẫn...")
# Hạ trần giá nhà xuống 200 triệu/m2 để loại bỏ hoàn toàn các điểm dị biệt gây bùng nổ MSE
df = df[(df['Diện tích'] >= 10) & (df['Diện tích'] <= 500)]
df = df[(df['Gia trieu/m2'] >= 15) & (df['Gia trieu/m2'] <= 200)] 

Q1 = df['Diện tích'].quantile(0.05)
Q3 = df['Diện tích'].quantile(0.95)
IQR = Q3 - Q1
df = df[(df['Diện tích'] >= (Q1 - 1.5 * IQR)) & (df['Diện tích'] <= (Q3 + 1.5 * IQR))]

# ==========================================
# 3. [KỸ THUẬT 2 & 3] GEO CLUSTERING & BINNING
# ==========================================
print("[2/4] Đang áp dụng Geo Clustering, Binning và Target Encoding...")

# Xử lý Thời gian
df['Ngày'] = pd.to_datetime(df['Ngày'])
df['Nam_Ban'] = df['Ngày'].dt.year
df['Thang_Ban'] = df['Ngày'].dt.month
df = df.drop(columns=['Ngày'])

# Target Encoding cho Phường/Xã
phuong_mean_price = df.groupby('Huyện')['Gia trieu/m2'].mean()
df['Phường (Giá nhà trung bình)'] = df['Huyện'].map(phuong_mean_price)
df = df.drop(columns=['Huyện'])

# Geo Clustering (Gom cụm Quận)
def map_tier(quan_name):
    quan_name = str(quan_name).lower()
    if any(q in quan_name for q in ['hoàn kiếm', 'ba đình']):
        return 'Tier 1 (Lõi di sản)'
    elif any(q in quan_name for q in ['đống đa', 'hai bà trưng', 'cầu giấy', 'thanh xuân', 'tây hồ']):
        return 'Tier 2 (Nội đô)'
    elif any(q in quan_name for q in ['hoàng mai', 'hà đông', 'long biên', 'nam từ liêm', 'bắc từ liêm']):
        return 'Tier 3 (Vùng ven)'
    else:
        return 'Tier 4 (Ngoại thành)'

df['Phan_Hang_Dia_Ly'] = df['Quận'].apply(map_tier)
df = df.drop(columns=['Quận'])

# Binning (Rời rạc hóa Số tầng)
def map_so_tang(tang):
    if tang <= 2:
        return 'Nhà thấp tầng (1-2)'
    elif tang <= 5:
        return 'Nhà tiêu chuẩn (3-5)'
    else:
        return 'Nhà cao tầng (>5)'

df['Nhom_So_Tang'] = df['Số tầng'].apply(map_so_tang)
df = df.drop(columns=['Số tầng'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Phan_Hang_Dia_Ly', 'Loại hình nhà ở', 'Nhom_So_Tang'], drop_first=True)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# ==========================================
# 4. [KỸ THUẬT 4] BIẾN ĐỔI LOGARIT (LOG TRANSFORMATION)
# ==========================================
print("[3/4] Đang áp dụng Log Transformation...")

# Chuyển đổi Mục tiêu (Giá) và Đặc trưng lệch phải (Diện tích) sang Log
df['Log_Gia'] = np.log1p(df['Gia trieu/m2'])
df['Log_Dien_tich'] = np.log1p(df['Diện tích'])

# Phân tách X và y (Dùng bản Log, bỏ bản gốc)
y = df['Log_Gia']
X = df.drop(columns=['Gia trieu/m2', 'Log_Gia', 'Diện tích'])
X['Log_Dien_tich'] = df['Log_Dien_tich']

# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("[4/4] Đang huấn luyện mô hình Linear Regression...\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ==========================================
# 6. DỰ BÁO & BIẾN ĐỔI NGƯỢC (INVERSE TRANSFORM)
# ==========================================
# Dự báo ra kết quả dạng Log
y_train_pred_log = model.predict(X_train_scaled)
y_test_pred_log = model.predict(X_test_scaled)

# [QUAN TRỌNG] Đưa giá trị Log về lại giá trị Thực tế (triệu/m2) bằng np.expm1
y_train_real = np.expm1(y_train)
y_train_pred_real = np.expm1(y_train_pred_log)

y_test_real = np.expm1(y_test)
y_test_pred_real = np.expm1(y_test_pred_log)

# ==========================================
# 7. ĐÁNH GIÁ MÔ HÌNH TRÊN ĐƠN VỊ THỰC TẾ
# ==========================================
# ----- TRAIN -----
train_mae = mean_absolute_error(y_train_real, y_train_pred_real)
train_mse = mean_squared_error(y_train_real, y_train_pred_real)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_real, y_train_pred_real)

# ----- TEST -----
test_mae = mean_absolute_error(y_test_real, y_test_pred_real)
test_mse = mean_squared_error(y_test_real, y_test_pred_real)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_real, y_test_pred_real)

# ----- CV(RMSE) -----
y_mean = y_test_real.mean()
cv_rmse = (test_rmse / y_mean) * 100

print("=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ĐƠN VỊ: TRIỆU/M²)")
print("=" * 60)

print(f"MAE (train):  {train_mae:.4f}")
print(f"MAE (test):   {test_mae:.4f}")
print()
print(f"MSE (train):  {train_mse:.4f}")
print(f"MSE (test):   {test_mse:.4f}")
print()
print(f"RMSE (train): {train_rmse:.4f}")
print(f"RMSE (test):  {test_rmse:.4f}")
print()
print(f"R² (train):   {train_r2:.4f}")
print(f"R² (test):    {test_r2:.4f}")
print()
print(f"CV(RMSE):     {cv_rmse:.2f}%")
print("-" * 60)

if cv_rmse <= 15:
    print("-> [XUẤT SẮC] Mô hình đạt tiêu chuẩn ASHRAE Guideline 14 (<15%)")
elif cv_rmse <= 20:
    print("-> [ĐẠT CHUẨN] Mô hình đạt tiêu chuẩn IPMVP (<20%)")
else:
    print("-> [CẦN CẢI THIỆN] Sai số còn cao, nên thử Random Forest, XGBoost,...")

print("=" * 60)

# ==========================================
# 8. KIỂM TRA OVERFITTING
# ==========================================
r2_gap = abs(train_r2 - test_r2)
print("\nKIỂM TRA OVERFITTING")
print("-" * 60)
print(f"Chênh lệch R² train-test: {r2_gap:.4f}")
if r2_gap < 0.05:
    print("-> Không có dấu hiệu overfitting đáng kể.")
elif r2_gap < 0.10:
    print("-> Có dấu hiệu overfitting nhẹ.")
else:
    print("-> Overfitting rõ rệt, cần cải thiện mô hình.")

# ==========================================
# 9. LƯU & VẼ BIỂU ĐỒ (ACTUAL VS PREDICTED)
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test_real, y_test_pred_real, alpha=0.5, color='blue')

min_val = min(y_test_real.min(), y_test_pred_real.min())
max_val = max(y_test_real.max(), y_test_pred_real.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=2)

plt.xlabel("Giá Thực Tế (triệu/m²)")
plt.ylabel("Giá Dự Báo (triệu/m²)")
plt.title("So Sánh Giá Thực Tế và Giá Dự Báo")
plt.tight_layout()

# Lưu ảnh trước, hiện biểu đồ sau
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# ==========================================
# 10. XUẤT KẾT QUẢ RA FILE
# ==========================================
results = pd.DataFrame({
    'Actual': y_test_real.values,
    'Predicted': y_test_pred_real
})
results['Error'] = results['Actual'] - results['Predicted']
results.to_csv('predictions.csv', index=False)

metrics = pd.DataFrame({
    'MAE_train': [train_mae],
    'MAE_test': [test_mae],
    'MSE_train': [train_mse],
    'MSE_test': [test_mse],
    'RMSE_train': [train_rmse],
    'RMSE_test': [test_rmse],
    'R2_train': [train_r2],
    'R2_test': [test_r2],
    'R2_gap': [r2_gap],
    'CV_RMSE_test': [cv_rmse]
})
metrics.to_csv('metrics.csv', index=False)

print("\n[MẶT ĐỐI MẶT] So sánh 5 căn nhà đầu tiên")
comparison_df = pd.DataFrame({
    'Giá Thực Tế': y_test_real.values[:5],
    'Giá Dự Báo': y_test_pred_real[:5]
})
comparison_df['Độ Lệch'] = comparison_df['Giá Thực Tế'] - comparison_df['Giá Dự Báo']
print(comparison_df.round(2))