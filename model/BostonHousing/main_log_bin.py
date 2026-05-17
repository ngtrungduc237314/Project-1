import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load data
# =========================
data = pd.read_csv('BostonHousing.csv')

print("First 5 rows:")
print(data.head())

# =========================
# 2. Feature Engineering
# =========================

# ---------------------------------
# (A) Tạo biến cờ cho price = 50
# ---------------------------------

# Boston Housing thường bị capped ở 50
# Nếu giá >= 50 -> 1
# Ngược lại -> 0

data['price_50_flag'] = np.where(data['price'] >= 50, 1, 0)

print("\nPrice_50_flag counts:")
print(data['price_50_flag'].value_counts())

# ---------------------------------
# (B) Log Transform cho PRICE
# ---------------------------------

# log(1 + price)
data['price_log'] = np.log1p(data['price'])

print("\nOriginal price statistics:")
print(data['price'].describe())

print("\nLog-transformed price statistics:")
print(data['price_log'].describe())

# ---------------------------------
# (C) Binning cho PRICE
# ---------------------------------

# Chia thành 4 nhóm giá
# low, medium, high, luxury

price_bins = [0, 15, 30, 45, np.inf]

price_labels = [
    'Low',
    'Medium',
    'High',
    'Luxury'
]

data['price_bin'] = pd.cut(
    data['price'],
    bins=price_bins,
    labels=price_labels
)

print("\nPrice bin counts:")
print(data['price_bin'].value_counts())

# One-hot encoding cho price_bin
data = pd.get_dummies(
    data,
    columns=['price_bin'],
    drop_first=True
)

# =========================
# 3. Feature & target
# =========================

# Target dùng price_log
y = data['price_log']

# Bỏ target gốc và target log khỏi X
X = data.drop(
    ['price', 'price_log'],
    axis=1
)

feature_names = X.columns

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# =========================
# 5. Standardize
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 6. Train model
# =========================
model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# 7. Predict
# =========================

# Predict trên log scale
y_pred_log = model.predict(X_test)
y_train_pred_log = model.predict(X_train)

# Chuyển ngược về scale gốc
y_pred = np.expm1(y_pred_log)
y_train_pred = np.expm1(y_train_pred_log)

# Actual values trên scale gốc
y_test_actual = np.expm1(y_test)
y_train_actual = np.expm1(y_train)

# =========================
# 8. Evaluate
# =========================

# ----- TRAIN -----
train_mse = mean_squared_error(
    y_train_actual,
    y_train_pred
)

train_r2 = r2_score(
    y_train_actual,
    y_train_pred
)

# ----- TEST -----
test_mse = mean_squared_error(
    y_test_actual,
    y_pred
)

test_r2 = r2_score(
    y_test_actual,
    y_pred
)

print("\n===== MODEL PERFORMANCE =====")

print(f"MSE (train): {train_mse:,.2f}")
print(f"MSE (test): {test_mse:,.2f}")

print(f"R² (train): {train_r2:.3f}")
print(f"R² (test): {test_r2:.3f}")

# =========================
# 9. Coefficients
# =========================
coeff_df = pd.DataFrame(
    model.coef_,
    feature_names,
    columns=['Coefficient']
)

coeff_df = coeff_df.sort_values(
    by='Coefficient',
    ascending=False
)

print("\n===== MODEL COEFFICIENTS =====")
print(coeff_df)

# =========================
# 10. Plot Actual vs Predicted
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(y_test_actual, y_pred)

# Đường y = x
min_val = min(
    y_test_actual.min(),
    y_pred.min()
)

max_val = max(
    y_test_actual.max(),
    y_pred.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# =========================
# 11. Histogram Before/After Log
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['price'], kde=True)
plt.title("Original Price")

plt.subplot(1, 2, 2)
sns.histplot(data['price_log'], kde=True)
plt.title("Log Transformed Price")

plt.tight_layout()

plt.savefig(
    "price_log_transform.png",
    dpi=300
)

plt.show()

# =========================
# 12. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test_actual.values,
    'Predicted': y_pred
})

results['Error'] = (
    results['Actual'] - results['Predicted']
)

results.to_csv(
    'predictions.csv',
    index=False
)

# =========================
# 13. Save metrics
# =========================
metrics = pd.DataFrame({
    'MSE_train': [train_mse],
    'MSE_test': [test_mse],
    'R2_train': [train_r2],
    'R2_test': [test_r2]
})

metrics.to_csv(
    'metrics.csv',
    index=False
)

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
print("- price_log_transform.png")

# data leakage do bien price_50_flag có thể giúp model biết được trường hợp đặc biệt, nên giữ lại biến này trong model để tránh data leakage
# hoac bien price_bin_Luxury có thể giúp model biết được trường hợp đặc biệt, nên giữ lại biến này trong model để tránh data leakage