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
print(data.head())

# =========================
# 2. Encode categorical
# =========================


# =========================
# 3. Feature & target
# =========================
X = data.drop('price', axis=1)

y = data['price']

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
# 6. Train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 7. Predict
# =========================
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

# =========================
# 8. Evaluate (đúng thứ tự yêu cầu)
# =========================
# --- TRAIN ---
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# --- TEST ---
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"MSE (train): {train_mse:,.2f}")
print(f"MSE (test): {test_mse:,.2f}")
print(f"R² (train): {train_r2:.3f}")
print(f"R² (test): {test_r2:.3f}")

# =========================
# 9. Coefficients
# =========================
coeff_df = pd.DataFrame(model.coef_, feature_names, columns=['Coefficient'])
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("\nModel Coefficients:")
print(coeff_df)

# =========================
# 10. Plot Actual vs Predicted + y = x
# =========================
plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (with y = x)")

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 11. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

results['Error'] = results['Actual'] - results['Predicted']
results.to_csv('predictions.csv', index=False)

# =========================
# 12. Save metrics (đúng thứ tự yêu cầu)
# =========================
metrics = pd.DataFrame({
    'MSE_train': [train_mse],
    'MSE_test': [test_mse],
    'R2_train': [train_r2],
    'R2_test': [test_r2]
})

metrics.to_csv('metrics.csv', index=False)

# =========================
# 13. Save coefficients
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