import os
import pandas as pd
import numpy as np

# ==================================================
# 1. CREATE OUTPUT FOLDER
# ==================================================

os.makedirs("validation_results", exist_ok=True)

# ==================================================
# 2. READ DATA
# ==================================================

df = pd.read_csv("Housing.csv")

print("=" * 70)
print("HOUSE DATA QUALITY VALIDATION")
print("=" * 70)

print(f"\nDataset shape : {df.shape}")

# ==================================================
# 3. CREATE ERROR LIST
# ==================================================

errors = []

# ==================================================
# 4. CHECK MISSING VALUES
# ==================================================

print("\nChecking missing values...")

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage (%)":
        (df.isnull().mean() * 100).round(2)
})

for col in df.columns:

    missing_rows = df[df[col].isnull()]

    for idx in missing_rows.index:

        errors.append({
            "row_index": idx,
            "column": col,
            "invalid_value": "NULL",
            "error_type": "Missing Value"
        })

# ==================================================
# 5. CHECK EMPTY STRINGS
# ==================================================

print("Checking empty strings...")

for col in df.columns:

    invalid_rows = df[
        (~df[col].isnull()) &
        (df[col].astype(str).str.strip() == "")
    ]

    for idx in invalid_rows.index:

        errors.append({
            "row_index": idx,
            "column": col,
            "invalid_value": "EMPTY STRING",
            "error_type": "Empty String"
        })

# ==================================================
# 6. CHECK DUPLICATE ROWS
# ==================================================

print("Checking duplicate rows...")

duplicate_rows = df[
    df.duplicated()
]

for idx in duplicate_rows.index:

    errors.append({
        "row_index": idx,
        "column": "ALL_COLUMNS",
        "invalid_value": "DUPLICATE ROW",
        "error_type": "Duplicate Record"
    })

# ==================================================
# 7. CHECK NUMERIC DATA TYPES
# ==================================================

print("Checking numeric data types...")

numeric_columns = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

for col in numeric_columns:

    if col in df.columns:

        invalid_rows = df[
            pd.to_numeric(
                df[col],
                errors="coerce"
            ).isnull()
            &
            (~df[col].isnull())
        ]

        for idx in invalid_rows.index:

            errors.append({
                "row_index": idx,
                "column": col,
                "invalid_value": df.loc[idx, col],
                "error_type": "Invalid Numeric Format"
            })

# ==================================================
# 8. CHECK NEGATIVE VALUES
# ==================================================

print("Checking negative values...")

non_negative_columns = [
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

for col in non_negative_columns:

    if col in df.columns:

        invalid_rows = df[
            (~df[col].isnull()) &
            (df[col] < 0)
        ]

        for idx in invalid_rows.index:

            errors.append({
                "row_index": idx,
                "column": col,
                "invalid_value": df.loc[idx, col],
                "error_type": "Negative Value"
            })

# ==================================================
# 9. CHECK INTEGER FORMAT
# ==================================================

print("Checking integer format...")

integer_columns = [
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

for col in integer_columns:

    if col in df.columns:

        invalid_rows = df[
            (~df[col].isnull()) &
            (
                df[col] != df[col].astype(int)
            )
        ]

        for idx in invalid_rows.index:

            errors.append({
                "row_index": idx,
                "column": col,
                "invalid_value": df.loc[idx, col],
                "error_type": "Invalid Integer Format"
            })

# ==================================================
# 10. CHECK YES/NO COLUMNS
# ==================================================

print("Checking categorical yes/no columns...")

yes_no_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in yes_no_columns:

    if col in df.columns:

        invalid_rows = df[
            ~df[col].isin(["yes", "no"])
        ]

        for idx in invalid_rows.index:

            errors.append({
                "row_index": idx,
                "column": col,
                "invalid_value": df.loc[idx, col],
                "error_type": "Invalid Yes/No Value"
            })

# ==================================================
# 11. CHECK FURNISHING STATUS
# ==================================================

print("Checking furnishing status...")

valid_furnishing = [
    "furnished",
    "semi-furnished",
    "unfurnished"
]

if "furnishingstatus" in df.columns:

    invalid_rows = df[
        ~df["furnishingstatus"].isin(valid_furnishing)
    ]

    for idx in invalid_rows.index:

        errors.append({
            "row_index": idx,
            "column": "furnishingstatus",
            "invalid_value": df.loc[idx, "furnishingstatus"],
            "error_type": "Invalid Furnishing Status"
        })

# ==================================================
# 12. CHECK INVALID RANGES
# ==================================================

print("Checking invalid ranges...")

range_rules = {
    "price": (0, 100000000),
    "area": (100, 100000),
    "bedrooms": (1, 20),
    "bathrooms": (1, 20),
    "stories": (1, 10),
    "parking": (0, 10)
}

for col, (min_val, max_val) in range_rules.items():

    if col in df.columns:

        invalid_rows = df[
            (~df[col].isnull()) &
            (
                (df[col] < min_val) |
                (df[col] > max_val)
            )
        ]

        for idx in invalid_rows.index:

            errors.append({
                "row_index": idx,
                "column": col,
                "invalid_value": df.loc[idx, col],
                "error_type": "Out Of Range"
            })

# ==================================================
# 13. CHECK OUTLIERS USING IQR
# ==================================================

print("Checking outliers using IQR...")

numeric_df = df.select_dtypes(include=np.number)

for col in numeric_df.columns:

    Q1 = numeric_df[col].quantile(0.25)
    Q3 = numeric_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_rows = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    for idx in outlier_rows.index:

        errors.append({
            "row_index": idx,
            "column": col,
            "invalid_value": df.loc[idx, col],
            "error_type": "Potential Outlier"
        })

# ==================================================
# 14. CREATE ERROR DATAFRAME
# ==================================================

errors_df = pd.DataFrame(errors)

# ==================================================
# 15. ERROR SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(f"\nTotal errors found: {len(errors_df)}")

if len(errors_df) > 0:

    print("\nError counts by type:\n")

    print(
        errors_df["error_type"]
        .value_counts()
    )

    print("\nFirst 20 errors:\n")

    print(errors_df.head(20))

else:

    print("\nNo errors found.")

# ==================================================
# 16. SAVE REPORTS
# ==================================================

errors_df.to_csv(
    "validation_results/house_data_quality_errors.csv",
    index=False
)

missing_summary.to_csv(
    "validation_results/house_missing_summary.csv"
)

print("\nFiles saved:")
print("1. validation_results/house_data_quality_errors.csv")
print("2. validation_results/house_missing_summary.csv")

# ==================================================
# 17. DATA TYPES
# ==================================================

print("\n" + "=" * 70)
print("COLUMN DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ==================================================
# 18. UNIQUE VALUE SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("UNIQUE VALUE SUMMARY")
print("=" * 70)

unique_summary = pd.DataFrame({
    "Unique Values": df.nunique()
})

print(unique_summary)

# ==================================================
# 19. DESCRIPTIVE STATISTICS
# ==================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())

# ==================================================
# 20. FINAL REPORT
# ==================================================

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print(f"""
Validation Completed
----------------------------------------
Rows Checked           : {df.shape[0]}
Columns Checked        : {df.shape[1]}
Total Errors           : {len(errors_df)}
Duplicate Rows         : {len(duplicate_rows)}
Rows With Missing Data : {df.isnull().any(axis=1).sum()}
Potential Outliers     : {
    len(errors_df[
        errors_df["error_type"] == "Potential Outlier"
    ])
}
""")

print("House dataset quality validation completed successfully.")