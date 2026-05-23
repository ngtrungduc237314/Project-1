import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 1. Read data
# ==================================================
def read_data():
    df = pd.read_csv("house_data.csv")
    return df


# ==================================================
# 2. Histogram
# ==================================================
def histogram():

    df = read_data()

    important_cols = [
        "price", "area", "bedrooms", "bathrooms",
        "stories", "parking"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(important_cols):

        sns.histplot(
            df[col],
            kde=True,
            bins=20,
            ax=axes[i],
            color='skyblue'
        )

        # Mean line
        axes[i].axvline(
            x=df[col].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label='Mean'
        )

        axes[i].set_title(
            f"Histogram of {col}",
            fontsize=11,
            fontweight='bold'
        )

        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

        axes[i].legend(fontsize=8)

    # Ẩn subplot dư
    for i in range(len(important_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Histogram of House Dataset Features",
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    fig.savefig(
        "histograms.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==================================================
# 3. Heatmap
# ==================================================
def heatmap():

    df = read_data()

    df_encoded = df.copy()

    binary_map = {
        "yes": 1,
        "no": 0
    }

    binary_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    for col in binary_cols:
        df_encoded[col] = df_encoded[col].map(binary_map)

    furnishing_map = {
        "furnished": 2,
        "semi-furnished": 1,
        "unfurnished": 0
    }

    df_encoded["furnishingstatus"] = df_encoded["furnishingstatus"].map(furnishing_map)

    corr_matrix = df_encoded.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        "Correlation Matrix of House Dataset",
        fontsize=15,
        fontweight='bold'
    )

    plt.savefig(
        "heatmap.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==================================================
# 4. Boxplot
# ==================================================
def boxplot():

    df = read_data()

    important_num_cols = [
        "price", "area", "bedrooms",
        "bathrooms", "stories", "parking"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(important_num_cols):

        sns.boxplot(
            data=df,
            y=col,
            ax=axes[i],
            color='skyblue'
        )

        axes[i].set_title(
            col,
            fontsize=12,
            fontweight='bold'
        )

        axes[i].tick_params(labelsize=9)

    for i in range(len(important_num_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        "Boxplot of House Dataset Features",
        fontsize=16,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    fig.savefig(
        "boxplot.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==================================================
# 5. Scatter Plot
# ==================================================
def scatter_plot():

    df = read_data()

    important_cols = [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(important_cols):

        sns.regplot(
            data=df,
            x=col,
            y="price",
            ax=axes[i],
            scatter_kws={"s": 20},
            line_kws={"color": "red"}
        )

        axes[i].set_title(
            f"{col} vs price",
            fontsize=11
        )

    for i in range(len(important_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Scatter Plot + Regression Line",
        fontsize=16,
        fontweight='bold'
    )

    fig.tight_layout()

    fig.savefig(
        "scatter.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==================================================
# 6. Missing Values
# ==================================================
def missing_values():

    df = read_data()

    missing = (df.isnull().sum() / len(df)) * 100

    missing = missing[missing > 0].sort_values(ascending=False)

    missing_data = pd.DataFrame({
        "Missing Ratio (%)": missing
    })

    print("\n===== MISSING VALUES =====")
    print(missing_data.head(10))

    if missing_data.empty:
        print("Dataset không có missing values.")

    else:

        plt.figure(figsize=(10, 6))

        missing.plot.barh(
            color='skyblue',
            edgecolor='black'
        )

        plt.title('Missing Data by Feature', fontsize=14)
        plt.xlabel('Missing Ratio (%)')
        plt.ylabel('Feature Name')

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.gca().invert_yaxis()

        plt.savefig(
            "missing_data.png",
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()


# ==================================================
# 7. Countplot (NEW)
# ==================================================
def countplot():

    df = read_data()

    categorical_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):

        sns.countplot(
            data=df,
            x=col,
            ax=axes[i],
            palette="Set2"
        )

        axes[i].set_title(
            f"Countplot of {col}",
            fontsize=11,
            fontweight='bold'
        )

        axes[i].tick_params(axis='x', rotation=20)

    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Countplot of Categorical Features",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    fig.savefig(
        "countplot.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==================================================
# 8. Main
# ==================================================
def main():

    df = read_data()

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== DATASET INFO =====")
    print(df.info())

    missing_values()

    scatter_plot()
    heatmap()
    boxplot()
    histogram()
    countplot()

    print("\nSaved files:")
    print("- scatter.png")
    print("- heatmap.png")
    print("- boxplot.png")
    print("- histograms.png")
    print("- countplot.png")


if __name__ == "__main__":
    main()