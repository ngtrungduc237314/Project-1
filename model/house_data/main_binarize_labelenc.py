import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================
# 1. Load data
# =========================
data = pd.read_csv('house_data.csv')

# =========================
# 2. Normalize price
# =========================
data['price'] = data['price'] / 1e6

print(data.head())

# =========================
# 3. Binarization (Yes/No → 0/1)
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
# 4. Label Encoding
# =========================
le = LabelEncoder()
data['furnishingstatus'] = le.fit_transform(data['furnishingstatus'])

# =========================
# 5. Feature & target
# =========================
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
# 7. Standardize
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 8. Train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 9. Predict
# =========================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================
# 10. Evaluate
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
# 11. Actual vs Predicted Plot
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], '--')

plt.xlabel("Actual Price (million)")
plt.ylabel("Predicted Price (million)")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 12. STANDARDIZED RESIDUAL PLOT (UPDATED)
# =========================

# residual
residuals = y_test - y_test_pred

# standard deviation of residuals
sigma = np.sqrt(mean_squared_error(y_test, y_test_pred))

# standardized residual
standardized_residuals = residuals / sigma

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, standardized_residuals)

plt.axhline(y=0, color='red', linestyle='--')
plt.axhline(y=2, color='gray', linestyle='--')
plt.axhline(y=-2, color='gray', linestyle='--')

plt.xlabel("Predicted Price (million)")
plt.ylabel("Standardized Residual")
plt.title("Standardized Residual Plot")

plt.tight_layout()
plt.savefig("standardized_residual_plot.png", dpi=300)
plt.show()

# =========================
# 13. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Residual': residuals,
    'Standardized_Residual': standardized_residuals
})

results['Error'] = results['Actual'] - results['Predicted']
results.to_csv('predictions.csv', index=False)

# =========================
# 14. Save metrics
# =========================
metrics = pd.DataFrame({
    'MSE_train': [MSE_train],
    'MSE_test': [MSE_test],
    'R2_train': [R2_train],
    'R2_test': [R2_test]
})

metrics.to_csv('metrics.csv', index=False)

# =========================
# DONE
# =========================
print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- actual_vs_predicted.png")
print("- standardized_residual_plot.png")