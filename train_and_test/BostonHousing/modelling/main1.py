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
# 2. Remove houses capped at 50
# =========================

before_rows = data.shape[0]

# Xóa các căn nhà bị ép giá trần = 50
data = data[data['price'] < 50]

after_rows = data.shape[0]

print("\n===== REMOVE PRICE CAPPED HOUSES =====")
print(f"Rows before: {before_rows}")
print(f"Rows after : {after_rows}")
print(f"Removed    : {before_rows - after_rows}")

# =========================
# 3. Feature Engineering
# =========================

# ----------------------------------
# 3.1 Log Transform cho PRICE
# ----------------------------------
data['price_log'] = np.log1p(data['price'])

# ----------------------------------
# 3.2 Log Transform cho CRIM và LSTAT
# ----------------------------------
data['crim_log'] = np.log1p(data['crim'])
data['lstat_log'] = np.log1p(data['lstat'])

# ----------------------------------
# 3.3 Binarization cho ZN
# ----------------------------------
# 1 nếu ZN > 0
# 0 nếu ZN = 0

data['zn_binary'] = np.where(data['zn'] > 0, 1, 0)

print("\n===== ZN BINARIZATION =====")
print(data['zn_binary'].value_counts())

# ----------------------------------
# 3.4 Trimming Outliers cho RM và PTRATIO
# ----------------------------------

def trim_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"\n{column}")
    print(f"Lower bound: {lower:.2f}")
    print(f"Upper bound: {upper:.2f}")

    before_rows = df.shape[0]

    # Xóa outlier
    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

    after_rows = df.shape[0]

    print(f"Removed rows: {before_rows - after_rows}")

    return df


# Trim RM
data = trim_outliers_iqr(data, 'rm')

# Trim PTRATIO
data = trim_outliers_iqr(data, 'ptratio')

print("\nDataset shape after trimming:")
print(data.shape)

# ----------------------------------
# 3.5 Xóa CRIM, LSTAT gốc,
#     TAX, RAD và ZN gốc
# ----------------------------------

# TAX và RAD xóa để giảm đa cộng tuyến
# ZN gốc xóa vì đã dùng zn_binary

data = data.drop(
    ['crim', 'lstat', 'tax', 'rad', 'zn'],
    axis=1
)

# ----------------------------------
# 3.6 Đổi tên biến log
# ----------------------------------

data = data.rename(columns={
    'crim_log': 'crim',
    'lstat_log': 'lstat'
})

print("\nColumns after feature engineering:")
print(data.columns)

# =========================
# 4. Feature & target
# =========================

X = data.drop(['price', 'price_log'], axis=1)

# Target log(price)
y = data['price_log']

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

# Predict trên scale log
y_pred_log = model.predict(X_test)
y_train_pred_log = model.predict(X_train)

# =========================
# 9. Inverse Transform
# =========================

# Chuyển về scale gốc
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

y_train_pred = np.expm1(y_train_pred_log)
y_train_original = np.expm1(y_train)

# =========================
# 10. Evaluate
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
# 11. Coefficients
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
# 12. Standardized Residual Plot
# =========================

# Residual thường
residuals = y_test_original - y_pred

# Số quan sát và số feature
n = len(y_test_original)
p = X_test.shape[1]

# Residual Standard Error (RSE)
RSE = np.sqrt(
    np.sum(residuals ** 2) / (n - p - 1)
)

# Standardized residuals
standardized_residuals = residuals / RSE

# Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    standardized_residuals,
    alpha=0.7
)

# Đường ngang tại 0
plt.axhline(
    y=0,
    linestyle='--',
    color='red'
)

# Ngưỡng ±2
plt.axhline(
    y=2,
    linestyle=':',
    color='orange'
)

plt.axhline(
    y=-2,
    linestyle=':',
    color='orange'
)

plt.xlabel("Predicted Price")
plt.ylabel("Standardized Residuals")

plt.title(
    "Standardized Residual Plot\n"
    "(Log + Trim + Remove TAX/RAD + ZN Binarization)"
)

plt.savefig(
    "standardized_residual_plot.png",
    dpi=300
)

plt.show()

# =========================
# 13. Save predictions
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

coeff_df.to_csv(
    'coefficients.csv'
)

# =========================
# DONE
# =========================

print("\nSaved files:")
print("- predictions.csv")
print("- metrics.csv")
print("- coefficients.csv")
print("- standardized_residual_plot.png")

"""
"ket hop log transform bien price, crim, lstat; trimming cho rm, ptratio; xoa bo bien tax, binarization bien zn, xoa bo bien rad.
"""