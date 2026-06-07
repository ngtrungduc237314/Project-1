import os
import pandas as pd
import numpy as np
import re

# ==========================================
# 0. THIẾT LẬP ĐƯỜNG DẪN ĐỒNG NHẤT
# ==========================================
input_path = '../dataset/HanoiHousing.csv'
output_path = '../dataset/HanoiHousing_Stage1.csv'

# 1. Đọc dữ liệu
df = pd.read_csv(input_path)

# Quét và xóa toàn bộ các cột có chữ "Unnamed" trong tên
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Xóa cột Địa chỉ
if 'Địa chỉ' in df.columns:
    df = df.drop(columns=['Địa chỉ'])

# ==========================================
# 2. ĐỔI TÊN CỘT TRƯỚC KHI XỬ LÝ
# ==========================================
df = df.rename(columns={'Giá/m2': 'Gia trieu/m2'})

print("--- TRƯỚC KHI XỬ LÝ ---")
print(df[['Diện tích', 'Gia trieu/m2', 'Số phòng ngủ', 'Dài', 'Rộng']].head(3))

# ==========================================
# 3. HÀM BÓC TÁCH SỐ THẦN THÁNH VÀ XỬ LÝ TIẾNG VIỆT
# ==========================================
def extract_float(text):
    # Nếu là ô trống (NaN), giữ nguyên
    if pd.isna(text):
        return np.nan
    
    # Ép về kiểu chuỗi, viết thường và xóa khoảng trắng 2 đầu
    text = str(text).lower().strip()
    
    # Xử lý ngoại lệ cá biệt của cột Số phòng ngủ
    if 'nhiều hơn 10' in text:
        return 11.0
    
    # Dùng Regex tìm chính xác cụm chứa số, dấu phẩy và dấu chấm
    match = re.search(r'[\d\,\.]+', text)
    if match:
        num_str = match.group(0)
        # Chuyển phẩy thập phân của tiếng Việt sang dấu chấm của Python
        num_str = num_str.replace(',', '.')
        try:
            return float(num_str)
        except ValueError:
            return np.nan
            
    return np.nan

# ==========================================
# 4. ÁP DỤNG HÀM VÀO CÁC CỘT CẦN LÀM SẠCH
# ==========================================
cols_to_clean = ['Diện tích', 'Gia trieu/m2', 'Số phòng ngủ', 'Dài', 'Rộng']

for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(extract_float)

# Ép kiểu số nguyên cho Số phòng ngủ và Số tầng
if 'Số phòng ngủ' in df.columns:
    df['Số phòng ngủ'] = df['Số phòng ngủ'].astype('Int64')

if 'Số tầng' in df.columns:
    df['Số tầng'] = df['Số tầng'].apply(extract_float).astype('Int64')

print("\n--- SAU KHI XỬ LÝ ---")
print(df[['Diện tích', 'Gia trieu/m2', 'Số phòng ngủ', 'Dài', 'Rộng']].head(3))

# ==========================================
# 5. LƯU FILE AN TOÀN
# ==========================================
# Tự động tạo thư mục chứa file đích nếu nó chưa tồn tại
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Lưu file dữ liệu đã làm sạch
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[THÀNH CÔNG] Đã lưu dữ liệu Giai đoạn 1 tại: {output_path}")