import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector # Đã thêm import

# =========================
# 1. Load data
# =========================
data = pd.read_csv('BostonHousing.csv')
# Fix tên cột tránh lỗi (nếu có)
data.columns = data.columns.str.strip().str.lower()
print(data.head())

# =========================
# 2. Encode categorical
# =========================


# =========================
# 3. Feature & target
# =========================
# Giả sử cột target tên là 'price' (hoặc 'medv' tùy format file của bạn)
target_col = 'price' if 'price' in data.columns else 'medv'
X = data.drop(target_col, axis=1)
y = data[target_col]

feature_names = X.columns

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# =========================
# 5. Standardize
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 5.5. FEATURE SELECTION (Forward Selection)
# =========================
# Khởi tạo mô hình gốc để dùng cho việc chọn biến
base_model = LinearRegression()

# Cấu hình Forward Selection
# Bạn có thể đổi n_features_to_select thành một con số cụ thể (ví dụ: 5) 
# Hoặc để 'auto' (kèm tol) để thuật toán tự tìm số lượng tối ưu dựa trên Cross-Validation
sfs = SequentialFeatureSelector(
    estimator=base_model, 
    n_features_to_select=5, # Chọn ra 5 biến tốt nhất. (Có thể đổi thành 'auto' ở các bản sklearn mới)
    direction='forward',
    cv=5 # Dùng 5-Fold CV để đánh giá biến
)

# Chạy thuật toán tìm biến trên tập Train
sfs.fit(X_train, y_train)

# Lấy ra danh sách tên các biến đã được chọn
selected_mask = sfs.get_support()
selected_features = feature_names[selected_mask]
print(f"\n[Feature Selection] Đã chọn {len(selected_features)} biến: {list(selected_features)}")

# Lọc lại tập Train và Test để vứt bỏ các biến không được chọn
X_train_sel = sfs.transform(X_train)
X_test_sel = sfs.transform(X_test)

# =========================
# 6. Train model (Chỉ train trên các biến đã chọn)
# =========================
model = LinearRegression()
model.fit(X_train_sel, y_train) # Dùng X_train_sel thay vì X_train

# =========================
# 7. Predict
# =========================
y_pred = model.predict(X_test_sel) # Dùng X_test_sel thay vì X_test

# =========================
# 8. Evaluate
# =========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.3f}")

# =========================
# 9. Overfitting check
# =========================
train_r2 = model.score(X_train_sel, y_train)
test_r2 = model.score(X_test_sel, y_test)

print("Train R2:", round(train_r2, 3))
print("Test R2:", round(test_r2, 3))

# =========================
# 10. Coefficients
# =========================
# Lưu ý: Lúc này model.coef_ chỉ có độ dài bằng số lượng biến đã chọn
coeff_df = pd.DataFrame(model.coef_, selected_features, columns=['Coefficient'])
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("\nModel Coefficients:")
print(coeff_df)

# =========================
# 11. Plot Actual vs Predicted + y = x
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted (Features: {len(selected_features)})")
plt.grid(True, linestyle='--', alpha=0.5)

# Lưu plot
plt.savefig("actual_vs_predicted.png", dpi=300)
# plt.show() # Uncomment để xem trực tiếp

# =========================
# 12. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

results['Error'] = results['Actual'] - results['Predicted']
results.to_csv('predictions.csv', index=False)

# =========================
# 13. Save metrics
# =========================
metrics = pd.DataFrame({
    'Num_Features': [len(selected_features)],
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'Train_R2': [train_r2],
    'Test_R2': [test_r2]
})

metrics.to_csv('metrics.csv', index=False)

# =========================
# 14. Save coefficients
# =========================
coeff_df.to_csv('coefficients.csv')

# =========================
# DONE
# =========================
print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- coefficients.csv")
print("- actual_vs_predicted.png")