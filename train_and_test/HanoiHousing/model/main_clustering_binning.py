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
# Phải lấy dữ liệu Stage 3 vì lúc này cột 'Quận' vẫn còn giữ nguyên dạng chữ
input_path = '../dataset/HanoiHousing_Stage3_Cleaned.csv'
df = pd.read_csv(input_path)

print("=" * 60)
print(f"ĐỌC DỮ LIỆU TỪ STAGE 3: {df.shape[0]} dòng | {df.shape[1]} cột")
print("=" * 60 + "\n")

# ==========================================
# [TÍCH HỢP] FEATURE ENGINEERING & GEOSPATIAL CLUSTERING
# ==========================================
print("[Đang xử lý] Áp dụng Feature Engineering & Phân hạng Địa lý...\n")

# 1. Lọc Ngoại lai (Outliers)
df = df[(df['Diện tích'] >= 10) & (df['Diện tích'] <= 500)]
df = df[(df['Gia trieu/m2'] >= 15) & (df['Gia trieu/m2'] <= 1000)]
Q1 = df['Diện tích'].quantile(0.05)
Q3 = df['Diện tích'].quantile(0.95)
IQR = Q3 - Q1
df = df[(df['Diện tích'] >= (Q1 - 1.5 * IQR)) & (df['Diện tích'] <= (Q3 + 1.5 * IQR))]

# 2. Xử lý Thời gian
df['Ngày'] = pd.to_datetime(df['Ngày'])
df['Nam_Ban'] = df['Ngày'].dt.year
df['Thang_Ban'] = df['Ngày'].dt.month
df = df.drop(columns=['Ngày'])

# 3. Target Encoding cho 'Huyện' (Phường/Xã)
phuong_mean_price = df.groupby('Huyện')['Gia trieu/m2'].mean()
df['Phường (Giá nhà trung bình)'] = df['Huyện'].map(phuong_mean_price)
df = df.drop(columns=['Huyện'])

# 4. [CỐT LÕI MỚI] PHÂN HẠNG CỤM ĐỊA LÝ (GEOSPATIAL CLUSTERING)
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

# Gán phân hạng và xóa cột Quận cũ
df['Phan_Hang_Dia_Ly'] = df['Quận'].apply(map_tier)
df = df.drop(columns=['Quận'])

# 5. One-Hot Encoding cho các biến phân loại còn lại
df = pd.get_dummies(df, columns=['Phan_Hang_Dia_Ly', 'Loại hình nhà ở'], drop_first=True)

# Ép toàn bộ cột True/False về 1/0
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# ==========================================
# 2. PHÂN TÁCH ĐẶC TRƯNG (X) VÀ MỤC TIÊU (y)
# ==========================================
y = df['Gia trieu/m2']
X = df.drop(columns=['Gia trieu/m2'])

# ==========================================
# 3. CHIA TRAIN / TEST
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Số lượng mẫu Huấn luyện (Train): {X_train.shape[0]} dòng")
print(f"Số lượng mẫu Kiểm thử (Test): {X_test.shape[0]} dòng\n")

# ==========================================
# 4. CHUẨN HÓA DỮ LIỆU
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH
# ==========================================
print("[Đang xử lý] Mô hình đang học...\n")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ==========================================
# 6. DỰ BÁO
# ==========================================
y_train_pred = model.predict(X_train_scaled)
y_pred = model.predict(X_test_scaled)

# ==========================================
# 7. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
# ----- TRAIN -----
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# ----- TEST -----
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

# ----- CV(RMSE) -----
y_mean = y_test.mean()
cv_rmse = (test_rmse / y_mean) * 100

print("=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
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
# 9. HỆ SỐ HỒI QUY
# ==========================================
coeff_df = pd.DataFrame(
    model.coef_,
    index=X.columns,
    columns=['Coefficient']
)
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("\nTOP HỆ SỐ HỒI QUY LỚN NHẤT")
print("-" * 60)
print(coeff_df.head(10))

# ==========================================
# 10. ACTUAL VS PREDICTED
# ==========================================
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')

plt.xlabel("Actual Price (triệu/m²)")
plt.ylabel("Predicted Price (triệu/m²)")
plt.title("Actual vs Predicted Price")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()
plt.close() # Đóng biểu đồ sau khi lưu để giải phóng bộ nhớ

# ==========================================
# LƯU KẾT QUẢ ĐẦU RA
# ==========================================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
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
coeff_df.to_csv('coefficients.csv')


print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- coefficients.csv")
print("- actual_vs_predicted.png")