import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =========================
# 1. Load data
# =========================
data = pd.read_csv('house_data.csv')

# =========================
# 2. Convert price
# =========================
# Đưa giá về đơn vị Triệu để dễ đọc, MSE sẽ tính theo (Triệu^2)
data['price'] = data['price'] / 1e6

# =========================
# 2.5 CAPPING OUTLIERS (Ép trần các biến đếm)
# =========================
# Dựa trên phân tích Boxplot trước đó
data['bedrooms'] = data['bedrooms'].clip(upper=4)
data['bathrooms'] = data['bathrooms'].clip(upper=2)
data['stories'] = data['stories'].clip(upper=3)
data['parking'] = data['parking'].clip(upper=2)

# =========================
# 3. Encode binary BEFORE interaction
# =========================
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# =========================
# 4. INTERACTION FEATURE (area × prefarea)
# =========================
# Tạo đặc trưng tương tác: Diện tích x Khu vực ưu tiên
data['area_x_prefarea'] = data['area'] * data['prefarea']

# =========================
# 5. Feature & target
# =========================
categorical_cols = [
    'furnishingstatus'
]

X = data.drop('price', axis=1)
y = data['price']

# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# =========================
# 7. Preprocessing (OneHot)
# =========================
# THÊM sparse_output=False ĐỂ TRÁNH LỖI KHI ĐƯA VÀO STANDARD SCALER
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# =========================
# 8. Standardize
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 9. Train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 10. Predict
# =========================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================
# 11. Evaluate
# =========================
MSE_train = mean_squared_error(y_train, y_train_pred)
R2_train = r2_score(y_train, y_train_pred)

MSE_test = mean_squared_error(y_test, y_test_pred)
R2_test = r2_score(y_test, y_test_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MSE_train: {MSE_train:,.2f}")
print(f"MSE_test : {MSE_test:,.2f}")
print(f"R2_train : {R2_train:.3f}")
print(f"R2_test  : {R2_test:.3f}")

# =========================
# 12. Actual vs Predicted
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')

plt.xlabel("Actual Price (million)")
plt.ylabel("Predicted Price (million)")
plt.title("Actual vs Predicted\n(Capping + Interaction + OneHot)")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 13. Standardized Residual Plot
# =========================
residuals = y_test - y_test_pred
sigma = np.sqrt(mean_squared_error(y_test, y_test_pred))
standardized_residuals = residuals / sigma

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, standardized_residuals, alpha=0.7)

plt.axhline(0, color='red', linestyle='--')
plt.axhline(2, color='gray', linestyle='--')
plt.axhline(-2, color='gray', linestyle='--')

plt.xlabel("Predicted Price (million)")
plt.ylabel("Standardized Residual")
plt.title("Standardized Residual Plot")

plt.tight_layout()
plt.savefig("standardized_residual_plot.png", dpi=300)
plt.show()

# =========================
# 14. Save results
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Residual': residuals,
    'Standardized_Residual': standardized_residuals
})

results.to_csv('predictions.csv', index=False)

metrics = pd.DataFrame({
    'MSE_train': [MSE_train],
    'MSE_test': [MSE_test],
    'R2_train': [R2_train],
    'R2_test': [R2_test]
})

metrics.to_csv('metrics.csv', index=False)

print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- actual_vs_predicted.png")
print("- standardized_residual_plot.png")