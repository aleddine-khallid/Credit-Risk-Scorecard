import pandas as pd
import numpy as np

def fine_class_variable(df, variable, target='GB_Flag', bins=10):
    """
    Apply fine classing to a numeric variable and return a summary table with counts and bad rate.
    """
    try:
        df_temp = df[[variable, target]].copy()

        # Replace placeholder values with NaN
        df_temp[variable] = df_temp[variable].replace([-9999998, -9999997.001, -9999, -1], np.nan)
        df_temp = df_temp.dropna()

        # Normalize and encode target: Good = 0, Bad = 1
        df_temp[target] = df_temp[target].str.strip().str.capitalize()
        df_temp[target] = df_temp[target].map({'Good': 0, 'Bad': 1})

        
        # Bin the variable
        df_temp['Bin'] = pd.qcut(df_temp[variable], q=bins, duplicates='drop')

        # Summary table
        summary = df_temp.groupby('Bin', observed=True).agg(
            count=(target, 'count'),
            Bads=(target, 'sum')
        ).reset_index()

        summary['Goods'] = summary['count'] - summary['Bads']
        summary['Bad rate(%)'] = (summary['Bads'] / summary['count'] * 100).round(2)

        summary = summary[['Bin', 'count', 'Goods', 'Bads', 'Bad rate(%)']]

        print(f"\n Fine classing summary for {variable}")
        return summary

    except Exception as e:
        print(f" Error in fine classing {variable}: {e}")
        return None


def fine_class_bulk(df, variable_list, target='GB_Flag', bins=10):
    """
    Apply fine classing to multiple variables and return a dictionary of summaries.
    """
    results = {}
    for var in variable_list:
        print(f" fine classing {var}")
        summary = fine_class_variable(df, variable=var, target=target, bins=bins)
        if summary is not None:
            results[var] = summary
    return results
    