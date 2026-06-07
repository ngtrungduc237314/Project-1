import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load data
# =========================
data = pd.read_csv('USA_Housing.csv', usecols=lambda col: col != 'Address')
print(data.head())

# =========================
# 2. Scale price (đơn vị trăm nghìn)
# =========================
data['price'] = data['price'] / 1e5

# =========================
# 3. CAPPING OUTLIERS (IQR method)
# =========================
cols_to_cap = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Area Population'
]

for col in cols_to_cap:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[col] = np.clip(data[col], lower_bound, upper_bound)

print("\nAfter capping:")
print(data[cols_to_cap].describe())

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
    X, y, test_size=0.25, random_state=0
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
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# =========================
# 9. Evaluate
# =========================
MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test = mean_squared_error(y_test, y_test_pred)

R2_train = r2_score(y_train, y_train_pred)
R2_test = r2_score(y_test, y_test_pred)

print(f"\nMSE_train: {MSE_train:,.2f}")
print(f"MSE_test : {MSE_test:,.2f}")
print(f"R2_train : {R2_train:.3f}")
print(f"R2_test  : {R2_test:.3f}")

# =========================
# 10. Coefficients
# =========================
coeff_df = pd.DataFrame(model.coef_, feature_names, columns=['Coefficient'])
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("\nModel Coefficients:")
print(coeff_df)

# =========================
# 11. Actual vs Predicted
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (After Capping)")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 12. STANDARDIZED RESIDUAL PLOT
# =========================
residuals = y_test - y_test_pred

sigma = np.sqrt(mean_squared_error(y_test, y_test_pred))
standardized_residuals = residuals / sigma

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, standardized_residuals)

plt.axhline(0, linestyle='--', color='red')
plt.axhline(2, linestyle='--', color='gray')
plt.axhline(-2, linestyle='--', color='gray')

plt.xlabel("Predicted Price")
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