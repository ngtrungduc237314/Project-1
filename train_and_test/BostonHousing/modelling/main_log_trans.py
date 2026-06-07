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
# Log Transform cho price
# =========================

# Dùng log1p để an toàn nếu có giá trị 0
data['price_log'] = np.log1p(data['price'])

print("\nOriginal price statistics:")
print(data['price'].describe())

print("\nLog-transformed price statistics:")
print(data['price_log'].describe())

# =========================
# 3. Feature & target
# =========================

# Bỏ price gốc và price_log khỏi feature
X = data.drop(['price', 'price_log'], axis=1)

# Target mới là price_log
y = data['price_log']

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

# Prediction trên scale log
y_pred_log = model.predict(X_test)
y_train_pred_log = model.predict(X_train)

# Chuyển ngược về scale gốc
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

y_train_pred = np.expm1(y_train_pred_log)
y_train_original = np.expm1(y_train)

# =========================
# 8. Evaluate
# =========================

# ----- TRAIN -----
train_mse = mean_squared_error(
    y_train_original,
    y_train_pred
)

train_r2 = r2_score(
    y_train_original,
    y_train_pred
)

# ----- TEST -----
test_mse = mean_squared_error(
    y_test_original,
    y_pred
)

test_r2 = r2_score(
    y_test_original,
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

plt.scatter(y_test_original, y_pred)

# Đường y = x
min_val = min(y_test_original.min(), y_pred.min())
max_val = max(y_test_original.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted (Log Transform)")

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()

# =========================
# 11. Distribution Comparison
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['price'], kde=True)
plt.title("Original Price Distribution")

plt.subplot(1, 2, 2)
sns.histplot(data['price_log'], kde=True)
plt.title("Log-Transformed Price Distribution")

plt.tight_layout()

plt.savefig("price_distribution_comparison.png", dpi=300)

plt.show()

# =========================
# 12. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test_original.values,
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
print("- price_distribution_comparison.png")