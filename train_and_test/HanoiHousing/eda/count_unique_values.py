import pandas as pd

# ==========================================
# ĐỌC DỮ LIỆU
# ==========================================
# Đọc file Giai đoạn 1 (đã dọn sạch NaN)
df = pd.read_csv('../dataset/HanoiHousing_Stage3_Cleaned.csv')

total_rows = len(df)

# Đặt tên file báo cáo đầu ra
output_filename = "unique_values_summary.txt"

# Mở file để ghi với chuẩn utf-8 để hiển thị đúng tiếng Việt
with open(output_filename, 'w', encoding='utf-8') as f:
    
    print("=" * 60, file=f)
    print(f" TỔNG QUAN DỮ LIỆU: {total_rows} dòng | {len(df.columns)} cột", file=f)
    print("=" * 60 + "\n", file=f)

    # ==========================================
    # QUÉT VÀ LIỆT KÊ GIÁ TRỊ TỪNG CỘT
    # ==========================================
    for col in df.columns:
        # Đếm số lượng giá trị duy nhất (tính cả NaN nếu có)
        unique_count = df[col].nunique(dropna=False)
        
        print(f"► CỘT: [{col}]", file=f)
        print(f"  Số loại giá trị khác nhau: {unique_count}", file=f)
        print("-" * 40, file=f)
        
        # KỊCH BẢN 1: Ít giá trị (Thường là biến Phân loại - Categorical)
        if unique_count <= 20:
            # In toàn bộ danh sách và tần suất xuất hiện
            counts = df[col].value_counts(dropna=False)
            for val, count in counts.items():
                # Tính phần trăm
                pct = (count / total_rows) * 100
                print(f"   • {val}: {count} dòng ({pct:.2f}%)", file=f)
                
        # KỊCH BẢN 2: Quá nhiều giá trị (Thường là biến Số - Numerical)
        else:
            print("   (Dữ liệu đa dạng -> Hiển thị 5 giá trị xuất hiện nhiều nhất)", file=f)
            counts = df[col].value_counts(dropna=False).head(5)
            for val, count in counts.items():
                # Tính phần trăm
                pct = (count / total_rows) * 100
                print(f"   • {val}: {count} dòng ({pct:.2f}%)", file=f)
                
            # Nếu là cột số (float/int), in thêm thống kê min - max
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"   => [Phạm vi giá trị: Từ {min_val} đến {max_val}]", file=f)
                
        print("=" * 60 + "\n", file=f)

# In thông báo ra màn hình terminal để bạn biết code đã chạy xong
print(f"[THÀNH CÔNG] Đã lưu báo cáo thống kê vào file: {output_filename}")