import pandas as pd

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv("BostonHousing.csv")

# =========================
# 2. Hiển thị thông tin cơ bản
# =========================
print("Kích thước dữ liệu:", df.shape)
print("\nCác cột trong dataset:")
print(df.columns.tolist())

# =========================
# 3. Tính giá trị trung bình của giá nhà
# =========================
mean_price = df["price"].mean()

# =========================
# 4. In kết quả
# =========================
print("\n===== KẾT QUẢ =====")
print(f"Giá nhà trung bình (đơn vị gốc): {mean_price:.4f}")

# Nếu price được lưu theo đơn vị 1000 USD
print(f"Giá nhà trung bình (USD): ${mean_price * 1000:,.2f}")

# =========================
# 5. Một số thống kê bổ sung
# =========================
print("\n===== THỐNG KÊ PRICE =====")
print(f"Giá nhỏ nhất : {df['price'].min():.4f}")
print(f"Giá lớn nhất : {df['price'].max():.4f}")
print(f"Trung vị     : {df['price'].median():.4f}")
print(f"Độ lệch chuẩn: {df['price'].std():.4f}")