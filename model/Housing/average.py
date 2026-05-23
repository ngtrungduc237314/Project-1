import pandas as pd

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_csv("Housing.csv")

# =========================k
# 2. Chuyển price sang đơn vị triệu
# =========================
df['price'] = df['price'] / 1e6

# =========================
# 3. Hiển thị thông tin cơ bản
# =========================
print("Kích thước dữ liệu:", df.shape)

print("\nCác cột trong dataset:")
print(df.columns.tolist())

# =========================
# 4. Tính giá nhà trung bình
# =========================
mean_price = df["price"].mean()

# =========================
# 5. In kết quả
# =========================
print("\n===== KẾT QUẢ =====")
print(f"Giá nhà trung bình: {mean_price:.4f} triệu")

# =========================
# 6. Một số thống kê bổ sung
# =========================
print("\n===== THỐNG KÊ PRICE =====")
print(f"Giá nhỏ nhất : {df['price'].min():.4f} triệu")
print(f"Giá lớn nhất : {df['price'].max():.4f} triệu")
print(f"Trung vị     : {df['price'].median():.4f} triệu")
print(f"Độ lệch chuẩn: {df['price'].std():.4f} triệu")