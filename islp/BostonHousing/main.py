import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
from ISLP.models import ModelSpec as MS
from ISLP.models import (
    Stepwise,
    sklearn_selection_path
)
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================
Boston = pd.read_csv('BostonHousing.csv')
Boston = Boston.dropna()

# Fix tên cột nếu có lỗi format
Boston.columns = Boston.columns.str.strip().str.lower()

print("Shape:", Boston.shape)

# =========================
# 2. TẠO X, Y
# =========================
design = MS(Boston.columns.drop('price')).fit(Boston)

Y = np.array(Boston['price'])
X = design.transform(Boston)
n = len(Y)

# =========================
# 3. FORWARD STEPWISE
# =========================
strategy = Stepwise.fixed_steps(
    design,
    len(design.terms),
    direction='forward'
)

full_path = sklearn_selection_path(OLS, strategy)
full_path.fit(Boston, Y)

# =========================
# 4. IN-SAMPLE
# =========================
Yhat_in = full_path.predict(Boston)

insample_mse = ((Yhat_in - Y[:, None])**2).mean(0)
n_steps = insample_mse.shape[0]

# =========================
# 5. CROSS-VALIDATION
# =========================
K = 5
kfold = skm.KFold(n_splits=K, shuffle=True, random_state=0)

Yhat_cv = skm.cross_val_predict(
    full_path,
    Boston,
    Y,
    cv=kfold
)

cv_mse = []
for train_idx, test_idx in kfold.split(Y):
    errors = (Yhat_cv[test_idx] - Y[test_idx, None])**2
    cv_mse.append(errors.mean(0))

cv_mse = np.array(cv_mse).T

cv_mean = cv_mse.mean(1)
cv_se = cv_mse.std(1) / np.sqrt(K)

# =========================
# 6. ONE-STANDARD-ERROR RULE
# =========================
best_idx = np.argmin(cv_mean)
min_cv = cv_mean[best_idx]
min_se = cv_se[best_idx]

threshold = min_cv + min_se

one_se_idx = np.where(cv_mean <= threshold)[0][0]

print("Best model (min CV): step =", best_idx)
print("One-SE model (simplest): step =", one_se_idx)

# =========================
# 6.5. LẤY BIẾN CỦA ONE-SE MODEL
# =========================

# số biến ở step one_se
p = one_se_idx + 1

# lấy p biến đầu tiên theo thứ tự stepwise
feature_names = [str(term) for term in design.terms[:p]]

print("Biến trong One-SE model:", feature_names)

with open("one_se_features.txt", "w") as f:
    for feat in feature_names:
        f.write(feat + "\n")

print("Đã lưu file one_se_features.txt")

# =========================
# 7. TÍNH RSE, R2
# =========================
results = []

TSS = np.sum((Y - Y.mean())**2)

for step in range(n_steps):
    yhat = Yhat_in[:, step]
    
    RSS = np.sum((Y - yhat)**2)
    p = step + 1
    
    RSE = np.sqrt(RSS / (n - p - 1))
    R2 = 1 - RSS / TSS
    
    results.append({
        "step": step,
        "in_sample_MSE": insample_mse[step],
        "cv_MSE": cv_mean[step],
        "RSE": RSE,
        "R2": R2
    })

results_df = pd.DataFrame(results)

# đánh dấu model
results_df["is_best"] = results_df["step"] == best_idx
results_df["is_one_se"] = results_df["step"] == one_se_idx

# =========================
# 8. LƯU FILE CSV
# =========================
results_df.to_csv("model_metrics.csv", index=False)
print("Đã lưu file model_metrics.csv")

# =========================
# 9. PLOT
# =========================
mse_fig, ax = subplots(figsize=(8, 8))

ax.plot(np.arange(n_steps), insample_mse, 'k', label='In-sample')

ax.errorbar(
    np.arange(n_steps),
    cv_mean,
    cv_se,
    color='r',
    label='Cross-validated'
)

# đánh dấu best model
ax.axvline(best_idx, color='blue', linestyle='--', label='Best (min CV)')

# đánh dấu one-SE model
ax.axvline(one_se_idx, color='green', linestyle='--', label='One-SE rule')

ax.set_xlabel('# steps')
ax.set_ylabel('MSE')
ax.legend()

# =========================
# 10. LƯU ĐỒ THỊ
# =========================
mse_fig.savefig("mse_plot.png", dpi=300, bbox_inches='tight')
print("Đã lưu file mse_plot.png")

plt.show()