import pandas as pd

# ==========================================
# 1. ĐỌC DỮ LIỆU GIAI ĐOẠN 4
# ==========================================
input_path = '../dataset/HanoiHousing_Stage4_Engineered.csv'
df = pd.read_csv(input_path)

# ==========================================
# 2. ĐẾM VÀ TÍNH PHẦN TRĂM
# ==========================================
# Đếm số lượng dòng theo từng năm
count_year = df['Nam_Ban'].value_counts()

# Tính tỷ lệ phần trăm (normalize=True)
pct_year = df['Nam_Ban'].value_counts(normalize=True) * 100

# Gộp lại thành một bảng DataFrame cho dễ xem
df_summary = pd.DataFrame({
    'Số lượng dòng': count_year,
    'Tỷ lệ (%)': pct_year.round(2)
})

# Sắp xếp theo thứ tự năm tăng dần
df_summary = df_summary.sort_index()

# ==========================================
# 3. HIỂN THỊ KẾT QUẢ
# ==========================================
print("=" * 45)
print("THỐNG KÊ SỐ LƯỢNG TIN ĐĂNG THEO NĂM")
print("=" * 45)
print(df_summary.to_string())
print("=" * 45)

# Nếu bạn muốn lưu bảng này ra file để đưa vào đồ án:
# df_summary.to_csv('Year_Distribution_Summary.csv')

print(df.groupby('Nam_Ban')['Gia trieu/m2'].describe())