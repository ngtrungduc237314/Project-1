import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================================================
# CONSTANTS
# ==================================================
PRICE_COL = 'Gia trieu/m2'
OUTPUT_DIR = 'eda_results_advanced'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_filename(text):
    return re.sub(r'[\\/*?:"<>| ]', '_', str(text))

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print("=" * 60)
    print("Dataset Shape:", df.shape)
    print("=" * 60)
    return df

def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

# ==================================================
# HISTOGRAM (GRID LAYOUT + ĐƯỜNG MEAN ĐỎ + ĐƯỜNG MODE XANH)
# ==================================================
def histogram(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_cols = 4 
    n_rows = math.ceil(len(numeric_cols) / n_cols) 
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() 
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        data_plot = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        sns.histplot(data=data_plot, x=col, kde=True, bins=40, color='lightblue', ax=ax)
        
        mean_val = data_plot[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        
        mode_series = data_plot[col].mode()
        if not mode_series.empty:
            mode_val = mode_series[0] 
            ax.axvline(mode_val, color='blue', linestyle='-.', linewidth=2, label='Mode')
        
        ax.set_title(f'Histogram of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Histogram of All Numeric Features', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/All_Histograms_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: All_Histograms_Grid.png")

# ==================================================
# SCATTERPLOT (GRID LAYOUT + ĐƯỜNG REGRESSION)
# ==================================================
def scatterplot(df):
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != PRICE_COL]
    n_cols = 4
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    price_upper = df[PRICE_COL].quantile(0.99)
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        col_lower = df[col].quantile(0.01)
        col_upper = df[col].quantile(0.99)
        
        data_plot = df[(df[col] >= col_lower) & (df[col] <= col_upper) & (df[PRICE_COL] <= price_upper)]

        sns.regplot(
            data=data_plot, 
            x=col, 
            y=PRICE_COL, 
            ax=ax,
            scatter_kws={'alpha': 0.6, 's': 15, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2} 
        )
        
        ax.set_title(f'{col} vs price', fontsize=12, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('price')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Scatter + Regression Line', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/All_Scatterplots_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: All_Scatterplots_Grid.png")

# ==================================================
# BOXPLOT (ĐÃ BỔ SUNG SẮP XẾP THỜI GIAN)
# ==================================================
def boxplot(df):
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    valid_cols = []
    for col in categorical_cols:
        if col != PRICE_COL and df[col].nunique() > 1:
            valid_cols.append(col)
            
    if not valid_cols:
        return

    n_cols = 4
    n_rows = math.ceil(len(valid_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 7 * n_rows))
    
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    price_upper = df[PRICE_COL].quantile(0.99)
    df_filtered_price = df[df[PRICE_COL] <= price_upper]

    for i, col in enumerate(valid_cols):
        ax = axes[i]

        if df_filtered_price[col].nunique() > 20:
            # Lấy top 20 phổ biến nhất (để hộp không đè nhau)
            top_cats = df_filtered_price[col].value_counts().nlargest(20).index
            data_plot = df_filtered_price[df_filtered_price[col].isin(top_cats)]
            title_suffix = "(Top 20 phổ biến)"
        else:
            data_plot = df_filtered_price
            title_suffix = ""

        # RẼ NHÁNH LOGIC SẮP XẾP
        if col in ['Ngày', 'Tháng', 'Năm']:
            # Biến thời gian -> Sắp xếp theo trình tự tăng dần từ cũ tới mới
            order = sorted([str(x) for x in data_plot[col].dropna().unique()])
        else:
            # Biến khác -> Sắp xếp theo Giá giảm dần
            order = data_plot.groupby(col)[PRICE_COL].median().sort_values(ascending=False).index

        sns.boxplot(data=data_plot, x=col, y=PRICE_COL, order=order, palette="viridis", ax=ax)
        
        ax.set_title(f'{PRICE_COL} theo {col} {title_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlabel('') 
        ax.set_ylabel(PRICE_COL)
        
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Boxplot of Categorical Features vs Price', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/All_Boxplots_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: All_Boxplots_Grid.png")

# ==================================================
# HEATMAP
# ==================================================
def heatmap(df):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Heatmap - Ma trận tương quan các biến số", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/heatmap_correlation.png", dpi=300)
    plt.close()
    print("Đã lưu: heatmap_correlation.png")

# ==================================================
# MAIN
# ==================================================
def main():
    df = load_data('../dataset/HanoiHousing_Stage3_Cleaned.csv')
    setup_plot_style()
    
    # ----------------------------------------------------
    # TIỀN XỬ LÝ CỘT THỜI GIAN ĐỂ VẼ BOXPLOT
    # ----------------------------------------------------
    if 'Ngày' in df.columns:
        # Chuyển đổi an toàn sang Datetime
        dt_col = pd.to_datetime(df['Ngày'], errors='coerce')
        
        # Ép cột Ngày gốc về chuỗi (String) để hàm boxplot nhận diện là biến phân loại
        df['Ngày'] = df['Ngày'].astype(str)
        
        # Sinh thêm cột Tháng (YYYY-MM) và Năm (YYYY) dưới dạng chuỗi
        df['Tháng'] = dt_col.dt.to_period('M').astype(str)
        df['Năm'] = dt_col.dt.year.fillna(0).astype(int).astype(str)
        
        # Dọn dẹp các giá trị rỗng bị ép kiểu sai
        df['Tháng'] = df['Tháng'].replace('NaT', np.nan)
        df['Năm'] = df['Năm'].replace('0', np.nan)
    # ----------------------------------------------------
    
    print("\n--- ĐANG VẼ HISTOGRAM GRID ---")
    histogram(df)
    print("\n--- ĐANG VẼ SCATTERPLOT GRID ---")
    scatterplot(df)
    print("\n--- ĐANG VẼ BOXPLOT GRID ---")
    boxplot(df)
    print("\n--- ĐANG VẼ HEATMAP ---")
    heatmap(df)
    
    print(f"\n[HOÀN TẤT] Báo cáo EDA đã được gói gọn hoàn hảo trong '{OUTPUT_DIR}'!")

if __name__ == "__main__":
    main()