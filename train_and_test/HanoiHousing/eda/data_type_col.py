import pandas as pd

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
# Bạn có thể thay đổi đường dẫn trỏ về file ban đầu 
file_path = '../dataset/HanoiHousing_Stage3_Cleaned.csv'
df = pd.read_csv(file_path)

print(f"Đang đọc dữ liệu từ: {file_path}")
print(f"Tổng số cột: {len(df.columns)}\n")

# ==========================================
# 2. LẤY KIỂU DỮ LIỆU VÀ CHUYỂN THÀNH DATAFRAME
# ==========================================
# df.dtypes trả về một Series, ta chuyển nó thành DataFrame cho đẹp
dtypes_df = pd.DataFrame({
    'Tên Cột': df.dtypes.index,
    'Kiểu Dữ Liệu': df.dtypes.values.astype(str) # Ép kiểu về chuỗi chữ để lưu CSV không bị lỗi
})

# In thử ra màn hình để kiểm tra nhanh
print("DANH SÁCH KIỂU DỮ LIỆU:")
print("-" * 40)
print(dtypes_df.to_string(index=False))
print("-" * 40)

# ==========================================
# 3. LƯU KẾT QUẢ RA FILE CSV
# ==========================================
output_filename = "Column_DataTypes.csv"

# Lưu ra file CSV (encoding='utf-8-sig' để Excel không bị lỗi font tiếng Việt)
dtypes_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n[THÀNH CÔNG] Đã lưu thông tin kiểu dữ liệu vào file: {output_filename}")