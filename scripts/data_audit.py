
import pandas as pd

def audit_basic_info(df):
  
    print("shape:", df.shape)

    print(" Data types:")
    print(df.dtypes)

    print("\nMissing values in %:")
    missing_perc = []
    for var in df.columns:
        perc = (df[var].isnull().sum() / df.shape[0]) * 100
        missing_perc.append(perc)

    missing_df = pd.DataFrame({
        'Variable': df.columns,
        'Missing %': missing_perc
    })

    print(missing_df[missing_df['Missing %'] > 0].sort_values(by="Missing %", ascending=False))
