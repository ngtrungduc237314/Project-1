import os
import pandas as pd

# ==========================================
# 0. THIẾT LẬP ĐƯỜNG DẪN
# ==========================================
input_path = '../dataset/HanoiHousing_Stage2.csv'
output_path = '../dataset/HanoiHousing_Stage3_Cleaned.csv'

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
df = pd.read_csv(input_path)

print("=" * 60)
print(f" TỔNG QUAN DỮ LIỆU (TRƯỚC KHI XÓA): {len(df)} dòng | {len(df.columns)} cột")
print("=" * 60 + "\n")

# ==========================================
# 2. XÓA CỘT 'Giấy tờ pháp lý' (Near-Zero Variance)
# ==========================================
if 'Giấy tờ pháp lý' in df.columns:
    df = df.drop(columns=['Giấy tờ pháp lý'])
    print(" [THÔNG BÁO] Đã xóa thành công cột 'Giấy tờ pháp lý'.\n")

total_rows = len(df)

# ==========================================
# 3. QUÉT VÀ LIỆT KÊ GIÁ TRỊ TỪNG CỘT (KÈM PHẦN TRĂM)
# ==========================================
for col in df.columns:
    unique_count = df[col].nunique(dropna=False)
    
    print(f"► CỘT: [{col}]")
    print(f"  Số loại giá trị khác nhau: {unique_count}")
    print("-" * 40)
    
    # KỊCH BẢN 1: Ít giá trị (Biến phân loại)
    if unique_count <= 20:
        counts = df[col].value_counts(dropna=False)
        for val, count in counts.items():
            pct = (count / total_rows) * 100
            print(f"   • {val}: {count} dòng ({pct:.2f}%)")
            
    # KỊCH BẢN 2: Quá nhiều giá trị (Biến số học)
    else:
        print("   (Dữ liệu đa dạng -> Hiển thị 5 giá trị xuất hiện nhiều nhất)")
        counts = df[col].value_counts(dropna=False).head(5)
        for val, count in counts.items():
            pct = (count / total_rows) * 100
            print(f"   • {val}: {count} dòng ({pct:.2f}%)")
            
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"   => [Phạm vi giá trị: Từ {min_val} đến {max_val}]")
            
    print("=" * 60 + "\n")

print(f" TỔNG QUAN DỮ LIỆU (SAU KHI XÓA): {len(df)} dòng | {len(df.columns)} cột\n")

# ==========================================
# 4. XUẤT FILE CSV VÀO FOLDER DATASET
# ==========================================
# Tự động tạo thư mục nếu chưa tồn tại (đề phòng lỗi đường dẫn)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Lưu ra file CSV (index=False để không in cột số thứ tự, utf-8-sig để chống lỗi font tiếng Việt)
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f" [THÀNH CÔNG] Đã lưu dữ liệu Giai đoạn 3 tại: {output_path}")