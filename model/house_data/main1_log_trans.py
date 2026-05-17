import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =========================
# 1. Load data
# =========================
data = pd.read_csv('house_data.csv')

# Chuyển price sang đơn vị TRIỆU VND
data['price'] = data['price'] / 1e6

# Log transform
data['price'] = np.log(data['price'])
data['area'] = np.log(data['area'])

print(data.head())

# =========================
# 2. Feature & target
# =========================
categorical_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

X = data.drop('price', axis=1)
y = data['price']

# =========================
# 3. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# =========================
# 4. Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

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
# 7. Predict (log -> exp)
# =========================
y_train_pred_log = model.predict(X_train)
y_test_pred_log = model.predict(X_test)

y_train_pred = np.exp(y_train_pred_log)
y_test_pred = np.exp(y_test_pred_log)

y_train_actual = np.exp(y_train)
y_test_actual = np.exp(y_test)

# =========================
# 8. Evaluate
# =========================
MSE_train = mean_squared_error(y_train_actual, y_train_pred)
R2_train = r2_score(y_train_actual, y_train_pred)

MSE_test = mean_squared_error(y_test_actual, y_test_pred)
R2_test = r2_score(y_test_actual, y_test_pred)

print("\n===== MODEL PERFORMANCE (million VND) =====")
print(f"MSE_train: {MSE_train:,.2f}")
print(f"MSE_test : {MSE_test:,.2f}")
print(f"R2_train : {R2_train:.3f}")
print(f"R2_test  : {R2_test:.3f}")

# =========================
# 9. Actual vs Predicted
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_test_pred, alpha=0.6)

min_val = min(y_test_actual.min(), y_test_pred.min())
max_val = max(y_test_actual.max(), y_test_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], '--', color='red')

plt.xlabel("Actual Price (million VND)")
plt.ylabel("Predicted Price (million VND)")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# =========================
# 10. Standardized Residual Plot
# =========================
residuals = y_test_actual - y_test_pred
sigma = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
standardized_residuals = residuals / sigma

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, standardized_residuals, alpha=0.6)

plt.axhline(0, linestyle='--', color='red')
plt.axhline(2, linestyle='--', color='gray')
plt.axhline(-2, linestyle='--', color='gray')

plt.xlabel("Predicted Price (million VND)")
plt.ylabel("Standardized Residual")
plt.title("Standardized Residual Plot")
plt.tight_layout()
plt.savefig("standardized_residual_plot.png", dpi=300)
plt.show()

# =========================
# 11. Save results
# =========================
results = pd.DataFrame({
    'Actual_million_VND': y_test_actual,
    'Predicted_million_VND': y_test_pred,
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