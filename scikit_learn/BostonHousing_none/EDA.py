import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data():
    # Đọc dữ liệu
    df = pd.read_csv("BostonHousing.csv")
    return df


# Bieu do cot (histogram) cho bien 'price'
def histogram():
    # Đọc dữ liệu
    df = read_data()

    # Vẽ histogram + KDE
    plt.figure(figsize=(8,5))
    sns.histplot(df["price"], kde=True)

    # Vẽ đường mean
    plt.axvline(x=df["price"].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

    # Label
    plt.title("Phân phối giá nhà (Price)")
    plt.xlabel("Price")
    plt.ylabel("Tần suất")
    plt.legend()

    # Luu file
    plt.savefig("histogram.png", dpi=300, bbox_inches='tight')

    plt.show()


# Heatmap tương quan giữa các biến
def heatmap(): 
    # Đọc dữ liệu
    df = read_data()

    # Tính ma trận tương quan
    corr_matrix = df.corr()

    # Vẽ heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

    # Title
    plt.title("Ma trận tương quan giữa các biến")

    # Lưu file
    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')

    plt.show()


# boxplot
def boxplot(): 
    # Đọc dữ liệu
    df = read_data()

    important_num_cols = [
        "crim", "zn", "indus", "nox", "rm",
        "age", "dis", "tax", "ptratio", "lstat", "price"
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18,8))
    axes = axes.flatten()

    for i, col in enumerate(important_num_cols):
        sns.boxplot(data=df, y=col, ax=axes[i], color='skyblue')
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].tick_params(labelsize=9)

    # Ẩn ô thừa
    for i in range(len(important_num_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Boxplot of Boston Housing Features", fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()

    # 👉 LƯU ẢNH
    fig.savefig("boxplot.png", dpi=300, bbox_inches='tight')

    plt.show()


# scatter plot 
def scatter_plot():
    # Đọc dữ liệu
    df = pd.read_csv("BostonHousing.csv")

    important_cols = ["crim", "zn", "indus", "nox", "rm",
                    "age", "dis", "tax", "ptratio", "lstat"]

    fig, axes = plt.subplots(3, 4, figsize=(18,10))
    axes = axes.flatten()

    for i, col in enumerate(important_cols):
        sns.regplot(data=df, x=col, y="price",
                    ax=axes[i],
                    scatter_kws={"s":15},
                    line_kws={"color": "red"})
        axes[i].set_title(f"{col} vs price")

    # Ẩn ô thừa
    for i in range(len(important_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Scatter + Regression Line", fontsize=16)
    fig.tight_layout()

    # 👉 Lưu ảnh
    fig.savefig("scatter.png", dpi=300, bbox_inches='tight')

    plt.show()


# Kiểm tra missing values
def missing_values():
    # Đọc dữ liệu
    df = pd.read_csv("BostonHousing.csv")

    # Tính tỷ lệ missing (%)
    missing = (df.isnull().sum() / len(df)) * 100

    # Lọc các cột có missing > 0
    missing = missing[missing > 0].sort_values(ascending=False)

    # Chuyển thành DataFrame
    missing_data = pd.DataFrame({"Missing Ratio (%)": missing})

    # In ra top 10 (nếu có)
    print(missing_data.head(10))

    # 👉 Nếu KHÔNG có missing thì báo luôn
    if missing_data.empty:
        print("Dataset không có missing values.")
    else:
        # Trực quan hoá
        plt.figure(figsize=(10, 6))
        missing.plot.barh(color='skyblue', edgecolor='black')

        plt.title('Missing Data by Feature', fontsize=14)
        plt.xlabel('Missing Ratio (%)')
        plt.ylabel('Feature Name')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()

        # 👉 Lưu ảnh
        plt.savefig("missing_data.png", dpi=300, bbox_inches='tight')

        plt.show()


def main():
    # Xem qua dữ liệu
    df = read_data()
    print(df.head())

    # Kiểm tra missing values
    missing_values()

    # Vẽ các biểu đồ EDA
    scatter_plot()
    heatmap()
    boxplot()
    histogram()
    
    
if __name__ == "__main__":
    main()