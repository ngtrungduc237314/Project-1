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
# 1. ĐỌC DỮ LIỆU ĐÃ QUA FEATURE ENGINEERING
# ==========================================
input_path = '../dataset/HanoiHousing_Stage4_Engineered.csv'
df = pd.read_csv(input_path)

print("=" * 60)
print(f"KHỞI ĐỘNG MÔ HÌNH VỚI DỮ LIỆU: {df.shape[0]} dòng | {df.shape[1]} cột")
print("=" * 60 + "\n")

# ==========================================
# 2. PHÂN TÁCH ĐẶC TRƯNG (X) VÀ MỤC TIÊU (y)
# ==========================================
# Log-transform biến mục tiêu
y = np.log1p(df['Gia trieu/m2'])

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
# 6. DỰ BÁO TRÊN THANG LOG
# ==========================================
y_train_pred_log = model.predict(X_train_scaled)
y_pred_log = model.predict(X_test_scaled)

# ==========================================
# 7. CHUYỂN VỀ ĐƠN VỊ GỐC
# ==========================================
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)

y_train_pred = np.expm1(y_train_pred_log)
y_pred = np.expm1(y_pred_log)

# ==========================================
# 8. ĐÁNH GIÁ MÔ HÌNH
# ==========================================

# ----- TRAIN -----
train_mae = mean_absolute_error(
    y_train_original,
    y_train_pred
)

train_mse = mean_squared_error(
    y_train_original,
    y_train_pred
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train_original,
    y_train_pred
)

# ----- TEST -----
test_mae = mean_absolute_error(
    y_test_original,
    y_pred
)

test_mse = mean_squared_error(
    y_test_original,
    y_pred
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test_original,
    y_pred
)

# ----- CV(RMSE) -----
y_mean = y_test_original.mean()
cv_rmse = (test_rmse / y_mean) * 100

print("=" * 60)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (LOG TRANSFORM)")
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
# 9. KIỂM TRA OVERFITTING
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
# 10. HỆ SỐ HỒI QUY
# ==========================================
coeff_df = pd.DataFrame(
    model.coef_,
    index=X.columns,
    columns=['Coefficient']
)

coeff_df = coeff_df.sort_values(
    by='Coefficient',
    ascending=False
)

print("\nTOP 10 HỆ SỐ HỒI QUY LỚN NHẤT")
print("-" * 60)
print(coeff_df.head(10))

# ==========================================
# 11. ACTUAL VS PREDICTED
# ==========================================
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_original,
    y_pred,
    alpha=0.6
)

min_val = min(
    y_test_original.min(),
    y_pred.min()
)

max_val = max(
    y_test_original.max(),
    y_pred.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--'
)

plt.xlabel("Actual Price (triệu/m²)")
plt.ylabel("Predicted Price (triệu/m²)")
plt.title("Actual vs Predicted (Log Transform)")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# ==========================================
# 12. LƯU DỰ BÁO
# ==========================================
results = pd.DataFrame({
    'Actual': y_test_original,
    'Predicted': y_pred
})

results['Error'] = (
    results['Actual']
    - results['Predicted']
)

results.to_csv(
    'predictions.csv',
    index=False
)

# ==========================================
# 13. LƯU METRICS
# ==========================================
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

metrics.to_csv(
    'metrics.csv',
    index=False
)

# ==========================================
# 14. LƯU HỆ SỐ
# ==========================================
coeff_df.to_csv(
    'coefficients.csv'
)


# ==========================================
# DONE
# ==========================================
print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- coefficients.csv")
print("- actual_vs_predicted.png")