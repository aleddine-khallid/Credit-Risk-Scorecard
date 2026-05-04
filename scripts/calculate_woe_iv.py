import numpy as np
import pandas as pd

def calculate_woe_iv(df, feature, target='GB_Flag'):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV)
    for a single coarse-classed variable.

    Parameters:
    - df: DataFrame containing the feature and target
    - feature: Name of the coarse-classed feature column (e.g., 'Application_Score_coarse')
    - target: Target variable name (expects 'Good'/'Bad')

    Returns:
    - woe_df: DataFrame with WoE calculation per bin (indexed by feature labels)
    - iv: Total Information Value (float)
    """
    df_temp = df[[feature, target]].copy()
    df_temp[target] = df_temp[target].map({'Good': 1, 'Bad': 0})

    grouped = df_temp.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Good']
    grouped['Bad'] = grouped['Total'] - grouped['Good']

# Distributions
    dist_good = (grouped['Good'] / grouped['Good'].sum()).replace(0, 1e-6)
    dist_bad = (grouped['Bad'] / grouped['Bad'].sum()).replace(0, 1e-6)

# WoE and IV
    grouped['WoE'] = np.log(dist_good / dist_bad)
    grouped['IV'] = (dist_good - dist_bad) * grouped['WoE']
    iv = grouped['IV'].sum()

# Return with the bin labels preserved as index
    return grouped[['Total', 'Good', 'Bad', 'WoE', 'IV']], iv

