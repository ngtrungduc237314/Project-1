import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Load data
# =========================
def read_data():
    df = pd.read_csv("USA_Housing.csv", usecols=lambda col: col != "Address")
    return df


# =========================
# 2. Histogram
# =========================
def histogram():
    df = read_data()

    cols = df.columns

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color='skyblue')

        axes[i].axvline(df[col].mean(), color='red', linestyle='--')
        axes[i].set_title(f"Histogram: {col}")

    plt.tight_layout()
    plt.savefig("histograms.png", dpi=300)
    plt.show()


# =========================
# 3. Boxplot
# =========================
def boxplot():
    df = read_data()

    cols = df.columns

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.savefig("boxplots.png", dpi=300)
    plt.show()


# =========================
# 4. Scatter plots (vs price)
# =========================
def scatter_plot():
    df = read_data()

    features = df.columns.drop("price")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.regplot(x=df[col], y=df["price"],
                    ax=axes[i],
                    scatter_kws={"s": 10},
                    line_kws={"color": "red"})
        axes[i].set_title(f"{col} vs Price")

    plt.tight_layout()
    plt.savefig("scatterplots.png", dpi=300)
    plt.show()


# =========================
# 5. Heatmap correlation
# =========================
def heatmap():
    df = read_data()

    plt.figure(figsize=(10, 6))
    corr = df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Correlation Heatmap")
    plt.savefig("heatmap.png", dpi=300)
    plt.show()


# =========================
# 6. MAIN
# =========================
def main():
    df = read_data()
    print(df.head())

    histogram()
    boxplot()
    scatter_plot()
    heatmap()


if __name__ == "__main__":
    main()