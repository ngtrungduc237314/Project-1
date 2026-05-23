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
# Binning + Flag variable
# =========================

# ----- Tạo biến cờ cho price = 50 -----
data['price_50_flag'] = np.where(data['price'] >= 50, 1, 0)

print("\nPrice = 50 Flag:")
print(data['price_50_flag'].value_counts())

# ----- Binning cho biến price -----
bins = [0, 20, 35, 50]

labels = ['Low', 'Medium', 'High']

data['price_category'] = pd.cut(
    data['price'],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("\nPrice categories:")
print(data['price_category'].value_counts())

# ----- One-hot encoding cho biến binning -----
price_dummies = pd.get_dummies(
    data['price_category'],
    prefix='price_bin'
)

data = pd.concat([data, price_dummies], axis=1)

# =========================
# 3. Feature & target
# =========================

# Không dùng price_category vì là text
# Không dùng price làm feature vì đó là target

X = data.drop(
    ['price', 'price_category'],
    axis=1
)

y = data['price']

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
y_pred = model.predict(X_test)

y_train_pred = model.predict(X_train)

# =========================
# 8. Evaluate
# =========================

# ----- TRAIN -----
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# ----- TEST -----
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

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

plt.title("Actual vs Predicted (with y = x)")

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.show()

# =========================
# 11. Visualization for bins
# =========================
plt.figure(figsize=(6, 4))

sns.countplot(x=data['price_category'])

plt.title("Distribution of Price Categories")

plt.savefig("price_bins_distribution.png", dpi=300)

plt.show()

# =========================
# 12. Save predictions
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
print("- price_bins_distribution.png")

# data leakage do bien price_50_flag có thể giúp model biết được trường hợp đặc biệt, nên giữ lại biến này trong model để tránh data leakage
# hoac bien price_bin_Luxury có thể giúp model biết được trường hợp đặc biệt, nên giữ lại biến này trong model để tránh data leakage