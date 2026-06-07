import os
import pandas as pd
import numpy as np

# ==========================================
# 0. THIẾT LẬP ĐƯỜNG DẪN ĐỒNG NHẤT
# ==========================================
input_path = '../dataset/HanoiHousing_Stage1.csv'
output_path = '../dataset/HanoiHousing_Stage2.csv'

# Đọc dữ liệu từ kết quả của Giai đoạn 1
df = pd.read_csv(input_path)

print("--- KIỂM TRA CỘT NÀO ĐANG CHỨA NaN ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("-" * 50)

# ==========================================
# BƯỚC 1: XÓA CÁC CỘT THIẾU QUÁ TRẦM TRỌNG
# ==========================================
columns_to_drop = ['Dài', 'Rộng']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# ==========================================
# BƯỚC 2: QUÉT SẠCH CÁC CỘT PHÂN LOẠI (TEXT)
# Điền bằng giá trị xuất hiện nhiều nhất (Mode)
# ==========================================
cat_cols = ['Giấy tờ pháp lý', 'Quận', 'Huyện', 'Loại hình nhà ở', 'Địa chỉ']
for col in cat_cols:
    if col in df.columns:
        # Tránh trường hợp cột toàn NaN không có mode
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# BƯỚC 3: XỬ LÝ CỘT SỐ NGUYÊN (SỐ TẦNG, SỐ PHÒNG NGỦ)
# Phải làm tròn (round) median để không bị lỗi Float -> Int64
# ==========================================
for col in ['Số tầng', 'Số phòng ngủ']:
    if col in df.columns:
        # Điền theo trung vị của từng Loại hình nhà ở
        df[col] = df.groupby('Loại hình nhà ở')[col].transform(
            lambda x: x.fillna(round(x.median()) if pd.notna(x.median()) else np.nan)
        )
        
        # Vét máng: Nếu vẫn còn NaN, dùng trung vị của toàn thành phố
        global_median = df[col].median()
        if pd.notna(global_median):
            df[col] = df[col].fillna(round(global_median))
        else:
            df[col] = df[col].fillna(1) # Backup đường cùng
            
        df[col] = df[col].astype('Int64')

# ==========================================
# BƯỚC 4: XỬ LÝ CỘT DIỆN TÍCH (SỐ THỰC)
# Diện tích nhà thường đồng đều theo Quận, nên tính Median theo Quận
# ==========================================
if 'Diện tích' in df.columns:
    df['Diện tích'] = df.groupby('Quận')['Diện tích'].transform(
        lambda x: x.fillna(x.median())
    )
    # Vét máng bằng trung vị toàn bộ dữ liệu
    df['Diện tích'] = df['Diện tích'].fillna(df['Diện tích'].median())

# ==========================================
# BƯỚC 5: XỬ LÝ BIẾN MỤC TIÊU (GIÁ NHÀ)
# ==========================================
# Nguyên tắc tối thượng của Hồi quy: KHÔNG TỰ ĐIỀN BIẾN MỤC TIÊU (Y).
# Nếu căn nhà không có giá, nó không có giá trị học tập cho thuật toán. -> XÓA DÒNG
if 'Gia trieu/m2' in df.columns:
    missing_target = df['Gia trieu/m2'].isnull().sum()
    if missing_target > 0:
        print(f" [Lưu ý ML] Đã xóa {missing_target} dòng do khuyết biến mục tiêu (Giá nhà).")
        df = df.dropna(subset=['Gia trieu/m2'])

# ==========================================
# KIỂM TRA LẠI LẦN CUỐI
# ==========================================
print("-" * 50)
print("--- TỔNG KẾT MISSING VALUES (SAU XỬ LÝ) ---")
missing_after = df.isnull().sum().sum()
if missing_after == 0:
    print(" Tuyệt vời! Bộ dữ liệu đã hoàn toàn đặc ruột, không còn một giá trị NaN nào.")
else:
    print(f" Vẫn còn {missing_after} giá trị NaN. Chi tiết:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# ==========================================
# LƯU LẠI FILE ĐÃ LÀM SẠCH Ở STAGE 2
# ==========================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[THÀNH CÔNG] Đã lưu dữ liệu Giai đoạn 2 tại: {output_path}")