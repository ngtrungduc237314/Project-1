import pandas as pd

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('..\dataset\HanoiHousing_Stage1.csv')

# 2. Đếm số lượng và tính tỷ lệ thiếu
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# 3. Gom kết quả vào một bảng DataFrame
missing_summary = pd.DataFrame({
    'Số lượng thiếu': missing_count,
    'Tỷ lệ thiếu (%)': missing_percentage
})

# 4. Lọc các cột có dữ liệu thiếu và sắp xếp giảm dần
missing_summary = missing_summary[missing_summary['Số lượng thiếu'] > 0]
missing_summary = missing_summary.sort_values(by='Tỷ lệ thiếu (%)', ascending=False)

# 5. [BƯỚC MỚI] Đưa tên cột (đang là index) thành một cột thực sự trong bảng
missing_summary = missing_summary.reset_index()
missing_summary = missing_summary.rename(columns={'index': 'Tên cột'})

# Làm tròn tỷ lệ phần trăm cho đẹp
missing_summary['Tỷ lệ thiếu (%)'] = missing_summary['Tỷ lệ thiếu (%)'].round(2)

# 6. [BƯỚC MỚI] Lưu ra file CSV
# Dùng encoding='utf-8-sig' để Excel không bị lỗi font tiếng Việt
missing_summary.to_csv('missing_values_summary.csv', index=False, encoding='utf-8-sig')

print("Đã lưu thành công danh sách các cột bị thiếu vào file 'missing_values_summary.csv'!")
print("\nBẢNG XEM TRƯỚC:")
print(missing_summary)