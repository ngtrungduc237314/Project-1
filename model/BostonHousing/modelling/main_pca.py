import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# 1. Load data
# =========================
data = pd.read_csv('BostonHousing.csv')

print(data.head())

# =========================
# 2. Feature & target
# =========================

# Tách riêng tax và rad
pca_features = data[['tax', 'rad']]

# Các feature còn lại
X = data.drop(['price', 'tax', 'rad'], axis=1)

y = data['price']

# =========================
# 3. Standardize tax & rad trước PCA
# =========================
pca_scaler = StandardScaler()

pca_scaled = pca_scaler.fit_transform(pca_features)

# =========================
# 4. PCA
# =========================
pca = PCA(n_components=1)

pca_component = pca.fit_transform(pca_scaled)

# Thêm feature PCA vào X
X['rad_tax_pca'] = pca_component

# Feature names
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
# 6. Standardize toàn bộ feature
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

# TRAIN
train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

# TEST
test_mse = mean_squared_error(
    y_test,
    y_pred
)

test_r2 = r2_score(
    y_test,
    y_pred
)

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

print("\nModel Coefficients:")
print(coeff_df)

# =========================
# 11. Plot Actual vs Predicted
# =========================
plt.figure(figsize=(7,6))

plt.scatter(
    y_test,
    y_pred
)

min_val = min(
    y_test.min(),
    y_pred.min()
)

max_val = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted (PCA)"
)

plt.savefig(
    "actual_vs_predicted_pca.png",
    dpi=300
)

plt.show()

# =========================
# 12. Save predictions
# =========================
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

results['Error'] = (
    results['Actual']
    - results['Predicted']
)

results.to_csv(
    'predictions_pca.csv',
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
    'metrics_pca.csv',
    index=False
)

# =========================
# 14. Save coefficients
# =========================
coeff_df.to_csv(
    'coefficients_pca.csv'
)

# =========================
# DONE
# =========================
print("\nSaved files:")
print("- predictions_pca.csv")
print("- metrics_pca.csv")
print("- coefficients_pca.csv")
print("- actual_vs_predicted_pca.png")