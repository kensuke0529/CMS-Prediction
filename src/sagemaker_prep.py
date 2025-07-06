import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

def sagemaker_prep(file_path):
    df = pd.read_parquet(file_path)

    # practice purpose, only 2023 
    df_2023 = df[df['year']==2023]

    df_2023 = df_2023.drop(columns=[
        'Rndrng_Prvdr_St', 'Rndrng_Prvdr_State_FIPS',
        'Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg', 'Avg_Tot_Pymt_Amt', 'year'
        ], axis=1)

    common_drgs = df_2023['DRG_Cd'].value_counts()[df_2023['DRG_Cd'].value_counts() > 25].index
    df_2023['drg_grouped'] = df_2023['DRG_Cd'].apply(lambda x: x if x in common_drgs else 'Other')

    cat_cols = ['Rndrng_Prvdr_CCN', 'DRG_Cd', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_Org_Name',
            'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn', 'RUCA_category', 'drg_grouped']

    # Encode each column
    for col in cat_cols:
        df_2023[col] = df_2023[col].astype('category')

    # change data format for sagemaker requirement
    target = 'Avg_Mdcr_Pymt_Amt'
    cols = [target] + [col for col in df_2023.columns if col != target]
    df_2023 = df_2023[cols]
    
    # 5% of random data
    df_sampled = df_2023.sample(frac=0.05, random_state=42)


    # Split into train/val/test
    train_val, test = train_test_split(df_sampled, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)
    
    base = Path().resolve()
    train.to_csv(base / 'data'/ 'train.csv', index=False, header=False)
    val.to_csv(base/'data'/"val.csv", index=False, header=False)
    test.to_csv(base/ 'data' / "test.csv", index=False, header=False)

def main():
    base = Path().resolve()
    file_path = base / "data" / "cms_cleaned.parquet"
    sagemaker_prep(file_path)

if __name__ == "__main__":
    main()