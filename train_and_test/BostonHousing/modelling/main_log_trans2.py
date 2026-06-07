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
# Log Transform cho CRIM và LSTAT
# =========================

# Dùng log1p để tránh lỗi log(0)
data['crim_log'] = np.log1p(data['crim'])
data['lstat_log'] = np.log1p(data['lstat'])

print("\nOriginal CRIM statistics:")
print(data['crim'].describe())

print("\nLog-transformed CRIM statistics:")
print(data['crim_log'].describe())

print("\nOriginal LSTAT statistics:")
print(data['lstat'].describe())

print("\nLog-transformed LSTAT statistics:")
print(data['lstat_log'].describe())

# =========================
# 3. Replace original features
# =========================

# Xóa biến gốc
data = data.drop(['crim', 'lstat'], axis=1)

# Đổi tên biến log cho dễ nhìn
data = data.rename(columns={
    'crim_log': 'crim',
    'lstat_log': 'lstat'
})

# =========================
# 4. Feature & target
# =========================
X = data.drop('price', axis=1)

y = data['price']

feature_names = X.columns

# =========================
# 5. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

# =========================
# 6. Standardize
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. Train model
# =========================
model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# 8. Predict
# =========================
y_pred = model.predict(X_test)

y_train_pred = model.predict(X_train)

# =========================
# 9. Evaluate
# =========================

# ----- TRAIN -----
train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

# ----- TEST -----
test_mse = mean_squared_error(
    y_test,
    y_pred
)

test_r2 = r2_score(
    y_test,
    y_pred
)

print("\n===== MODEL PERFORMANCE =====")

print(f"MSE (train): {train_mse:,.2f}")
print(f"MSE (test): {test_mse:,.2f}")

print(f"R² (train): {train_r2:.3f}")
print(f"R² (test): {test_r2:.3f}")

# =========================
# 10. Coefficients
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
# 11. Plot Actual vs Predicted
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

# Đường y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()

# =========================
# 12. Distribution Comparison
# =========================
plt.figure(figsize=(12, 5))

# ----- CRIM -----
plt.subplot(2, 2, 1)
sns.histplot(np.expm1(data['crim']), kde=True)
plt.title("Original CRIM")

plt.subplot(2, 2, 2)
sns.histplot(data['crim'], kde=True)
plt.title("Log-Transformed CRIM")

# ----- LSTAT -----
plt.subplot(2, 2, 3)
sns.histplot(np.expm1(data['lstat']), kde=True)
plt.title("Original LSTAT")

plt.subplot(2, 2, 4)
sns.histplot(data['lstat'], kde=True)
plt.title("Log-Transformed LSTAT")

plt.tight_layout()

plt.savefig("log_transform_comparison.png", dpi=300)

plt.show()

# =========================
# 13. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
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
# 14. Save metrics
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
# 15. Save coefficients
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
print("- log_transform_comparison.png")