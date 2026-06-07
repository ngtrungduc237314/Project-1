import pandas as pd
import numpy as np
import os

# ==========================================
# 0. THIẾT LẬP ĐƯỜNG DẪN
# ==========================================
input_path = '../dataset/HanoiHousing_Stage3_Cleaned.csv'
output_path = '../dataset/HanoiHousing_Stage4_Engineered.csv'

# 1. ĐỌC DỮ LIỆU GIAI ĐOẠN 3
df = pd.read_csv(input_path)
print(f"Kích thước ban đầu: {df.shape}")

# ==========================================
# BƯỚC 1: XỬ LÝ NGOẠI LAI (OUTLIERS)
# ==========================================
# Lọc bỏ các giá trị phi lý theo kiến thức thực tế
df = df[(df['Diện tích'] >= 10) & (df['Diện tích'] <= 500)]
df = df[(df['Gia trieu/m2'] >= 15) & (df['Gia trieu/m2'] <= 1000)]

# Gọt bớt các điểm dị biệt bằng phương pháp phân vị toán học (IQR)
Q1 = df['Diện tích'].quantile(0.05)
Q3 = df['Diện tích'].quantile(0.95)
IQR = Q3 - Q1
df = df[(df['Diện tích'] >= (Q1 - 1.5 * IQR)) & (df['Diện tích'] <= (Q3 + 1.5 * IQR))]

print(f"Kích thước sau khi cắt bỏ Ngoại lai: {df.shape}")

# ==========================================
# BƯỚC 2: KHAI PHÁ CỘT THỜI GIAN
# ==========================================
df['Ngày'] = pd.to_datetime(df['Ngày'])
df['Nam_Ban'] = df['Ngày'].dt.year
df['Thang_Ban'] = df['Ngày'].dt.month

# Xóa cột Ngày gốc vì thuật toán không đọc được định dạng Datetime
df = df.drop(columns=['Ngày'])

# ==========================================
# BƯỚC 3: MÃ HÓA BIẾN PHÂN LOẠI (ENCODING)
# ==========================================

# 3.1. Target Encoding cho cột 'Huyện' (Phường/Xã) và ĐỔI TÊN CỘT THEO YÊU CẦU
phuong_mean_price = df.groupby('Huyện')['Gia trieu/m2'].mean()
df['Phường (Giá nhà trung bình)'] = df['Huyện'].map(phuong_mean_price)

# Xóa cột Huyện dạng chữ gốc
df = df.drop(columns=['Huyện'])

# 3.2. One-Hot Encoding cho 'Quận' và 'Loại hình nhà ở'
# drop_first=True để tránh bẫy đa cộng tuyến (Dummy Variable Trap)
df = pd.get_dummies(df, columns=['Quận', 'Loại hình nhà ở'], drop_first=True)

# Ép toàn bộ các cột One-hot (True/False) về số (1/0)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# ==========================================
# 4. KIỂM TRA VÀ LƯU KẾT QUẢ
# ==========================================
print("=" * 60)
print(f"KÍCH THƯỚC DỮ LIỆU CUỐI CÙNG: {df.shape[0]} dòng | {df.shape[1]} cột")
print("=" * 60)

# Xem thử kết quả các cột quan trọng
print(df[['Diện tích', 'Nam_Ban', 'Thang_Ban', 'Phường (Giá nhà trung bình)']].head())

# Lưu file hoàn chỉnh vào folder dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n[THÀNH CÔNG] Dữ liệu đã sẵn sàng tại: {output_path}")