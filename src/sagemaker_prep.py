import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def sagemaker_prep(file_path):
    df = pd.read_parquet(file_path)

    # Filter 2023 only (practice purpose)
    df_2023 = df[df['year'] == 2023].copy()

    df_2023 = df_2023.drop(columns=[
        'Rndrng_Prvdr_St', 'Rndrng_Prvdr_State_FIPS',
        'Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg', 'Avg_Tot_Pymt_Amt', 'year'
    ], axis=1)

    # Group rare DRGs under 'Other'
    common_drgs = df_2023['DRG_Cd'].value_counts()[lambda x: x > 25].index
    df_2023['drg_grouped'] = df_2023['DRG_Cd'].apply(lambda x: x if x in common_drgs else 'Other')

    cat_cols = [
        'Rndrng_Prvdr_CCN', 'DRG_Cd', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_Org_Name',
        'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn', 'RUCA_category', 'drg_grouped'
    ]

    # Convert to category and save category mappings
    cat_encoders = {}
    for col in cat_cols:
        df_2023[col] = df_2023[col].astype('category')
        cat_encoders[col] = df_2023[col].cat.categories.tolist()  # save category list in order

    # Rearrange columns: target first
    target = 'Avg_Mdcr_Pymt_Amt'
    cols = [target] + [col for col in df_2023.columns if col != target]
    df_2023 = df_2023[cols]

    # Sample 5% data
    df_sampled = df_2023.sample(frac=0.05, random_state=42)

    # Split train, val, test
    train_val, test = train_test_split(df_sampled, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)

    base = Path().resolve().parent

    # Save CSVs WITH header (needed for proper column alignment during eval)
    train.to_csv(base / 'data' / 'train.csv', index=False)
    val.to_csv(base / 'data' / 'val.csv', index=False)
    test.to_csv(base / 'data' / 'test.csv', index=False)

    # Save categorical encoders for reuse later
    with open(base / 'data' / 'cat_encoders.json', 'w') as f:
        json.dump(cat_encoders, f)

    print("Data preparation done. Categorical encoders saved to 'cat_encoders.json'.")

def main():
    base = Path().resolve().parent
    file_path = base / "data" / "cms_cleaned.parquet"
    sagemaker_prep(file_path)

if __name__ == "__main__":
    main()
