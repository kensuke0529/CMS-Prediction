import pandas as pd
import joblib
from pathlib import Path

base = Path().resolve().parent 
model_path = base / 'model' / 'xgb_pipeline.joblib'  

model = joblib.load(model_path)

# samaple data and predict medicare payment 
sample = {
    'Rndrng_Prvdr_CCN': [390027],
    'DRG_Cd': ['291'],
    'RUCA_category': ['metro_core'],
    'Rndrng_Prvdr_State_FIPS': [42],      
    'Rndrng_Prvdr_Zip5': [19140],
    'drg_grouped': ['291'],
    'year': [2023],                         
}

test_df = pd.DataFrame(sample)

cat_cols = ['Rndrng_Prvdr_CCN', 'DRG_Cd', 'RUCA_category', 'Rndrng_Prvdr_State_FIPS',
            'Rndrng_Prvdr_Zip5', 'drg_grouped', 'year']

for col in cat_cols:
    test_df[col] = test_df[col].astype(str)

predictions = model.predict(test_df)

predictions = model.predict(test_df)

for i, pred in enumerate(predictions):
    print(f"Prediction {i+1}: ${pred:,.2f}")
