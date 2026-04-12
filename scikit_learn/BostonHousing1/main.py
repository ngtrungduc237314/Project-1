import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# =========================
# 8. Evaluate
# =========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.3f}")

# =========================
# 9. Overfitting check
# =========================
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("Train R2:", train_r2)
print("Test R2:", test_r2)

# =========================
# 10. Coefficients
# =========================
coeff_df = pd.DataFrame(model.coef_, feature_names, columns=['Coefficient'])
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("\nModel Coefficients:")
print(coeff_df)

# =========================
# 11. Plot Actual vs Predicted + y = x
# =========================
plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (with y = x)")

# Lưu plot
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 12. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

# thêm cột sai số (bonus)
results['Error'] = results['Actual'] - results['Predicted']

results.to_csv('predictions.csv', index=False)

# =========================
# 13. Save metrics
# =========================
metrics = pd.DataFrame({
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