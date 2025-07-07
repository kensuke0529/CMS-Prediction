import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
import numpy as np

def etl(file_path):
    
    df = pd.read_parquet(file_path)

    num_cols = ['Tot_Dschrgs', 'Avg_Submtd_Cvrd_Chrg',
            'Avg_Tot_Pymt_Amt', 'Avg_Mdcr_Pymt_Amt', 
            'Rndrng_Prvdr_State_FIPS', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_RUCA']

    df[num_cols] = df[num_cols].replace('', np.nan)
    df[num_cols] = df[num_cols].astype('float')


    ## Combine RUCA codes into shorter categories and if missing -> 'unknown'

    def categorize_ruca(desc):
        if pd.isna(desc) or desc == 'Unknown':
            return 'unknown'

        desc = str(desc)

        if desc.startswith('Metropolitan area core:'):
            return 'metro_core'
        elif desc.startswith('Metropolitan area high commuting:'):
            return 'metro_high_commute'
        elif desc.startswith('Metropolitan area low commuting:'):
            return 'metro_low_commute'

        elif desc.startswith('Micropolitan area core:'):
            return 'micro_core'
        elif desc.startswith('Micropolitan high commuting:'):
            return 'micro_high_commute'
        elif desc.startswith('Micropolitan low commuting:'):
            return 'micro_low_commute'

        elif desc.startswith('Small town core:'):
            return 'small_town_core'
        elif desc.startswith('Small town high commuting:'):
            return 'small_town_high_commute'
        elif desc.startswith('Small town low commuting:'):
            return 'small_town_low_commute'

        elif 'Secondary flow 30% to <50% to a larger urbanized area' in desc:
            return 'secondary_flow_metro'
        elif 'Secondary flow 30% to <50% to a urbanized area' in desc:
            return 'secondary_flow_metro'
        elif 'Secondary flow 30% to <50% to a urban cluster' in desc:
            return 'secondary_flow_micro'
        elif 'Secondary flow 30% to <50% to a urban cluster of 2,500 to 9,999' in desc:
            return 'secondary_flow_micro'

        elif desc.startswith('Rural areas:'):
            return 'rural'

        else:
            return 'unknown'

    df['RUCA_category'] = df['Rndrng_Prvdr_RUCA_Desc'].apply(categorize_ruca)

    df = df.drop(columns=['Rndrng_Prvdr_RUCA', 'DRG_Desc', 'Rndrng_Prvdr_RUCA_Desc'])
    
    return df 

def main():
    base = Path(__file__).resolve().parent.parent  
    file_path = base / "data" / "cms_raw.parquet"
    df = etl(file_path)
    
    (base / "data").mkdir(parents=True, exist_ok=True)
    
    cleaned_path = base / 'data' / "cms_cleaned.parquet"
    df.to_parquet(cleaned_path, index=False)
    print(f"Saved to {cleaned_path}")
    
if __name__ == "__main__":
    main()