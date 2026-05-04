import numpy as np
import pandas as pd
from scripts.data_loader import load_data
from scripts.data_audit import audit_basic_info
from scripts.subpopulation import filter_and_split
from scripts.fine_classing import fine_class_variable, fine_class_bulk
from scripts.coarse_classing import coarse_class_variable
from scripts.calculate_woe_iv import calculate_woe_iv
from scripts.good_woe_variables import select_variables_by_iv

def main():
    print(" Summer project pipeline \n")

# read and audit data
    df = load_data("data/BankCaseStudyData.csv")
    print(df.head())
    audit_basic_info(df)

# Subpopulation split
    dev_df, val_df = filter_and_split(df)

#Fine classing
    continuous_variables = [
        "loan_to_income", "Application_Score", "Gross_Annual_Income", "Loan_Amount",
        "Time_at_Address", "Time_in_Employment", "Time_with_Bank",
        "Age_of_Applicant", "Bureau_Score"
    ]
    fine_class_results = fine_class_bulk(dev_df, continuous_variables, target='GB_Flag', bins=6)

# Coarse classing mapping
    coarse_bin_dicts = {
        "Application_Score": {
            "High Risk": ['(554.999, 890.0]', '(890.0, 925.0]'],
            "Medium Risk": ['(925.0, 955.0]', '(955.0, 975.0]'],
            "Low Risk": ['(975.0, 985.0]', '(985.0, 1085.0]']
        },
        "loan_to_income": {
            'High Risk': ['(57.067, 11602.94]'],
            'Medium Risk': ['(24.67, 35.0]', '(35.0, 57.067]'],
            'Low Risk': ['(-9999997.001, 8.643]', '(8.643, 15.943]', '(15.943, 24.67]']
        },
        "Gross_Annual_Income": {
            'Very High Risk': ['(-0.001, 8000.0]'],
            'High Risk': ['(8000.0, 10500.0]'],
            'Medium Risk': ['(10500.0, 13000.0]', '(13000.0, 16200.0]'],
            'Low Risk': ['(16200.0, 22242.333]', '(22242.333, 420000.0]']
        },
        "Loan_Amount": {
            'Low Risk': ['(-0.001, 1000.0]'],
            'High Risk': ['(1000.0, 2150.0]', '(2150.0, 3500.0]'],
            'Medium-High Risk': ['(3500.0, 5000.0]', '(8212.667, 232100.0]'],
            'Medium Risk': ['(5000.0, 8212.667]']
        },
        "Time_at_Address": {
            'High Risk': ['(-0.001, 200.0]'],
            'Medium-High Risk': ['(200.0, 400.0]', '(1200.0, 2000.0]'],
            'Medium Risk': ['(400.0, 706.0]', '(706.0, 1200.0]', '(2000.0, 7500.0]']
        },
        "Time_in_Employment": {
            'High Risk': ['(-0.001, 100.0]', '(100.0, 300.0]'],
            'Medium Risk': ['(300.0, 600.0]', '(600.0, 1000.0]'],
            'Low Risk': ['(1000.0, 1700.0]', '(1700.0, 6100.0]']
        },
        "Time_with_Bank": {
            'High Risk': ['(-0.001, 203.0]', '(203.0, 406.0]'],
            'Medium Risk': ['(406.0, 702.0]', '(702.0, 1006.0]'],
            'Low Risk': ['(1006.0, 1206.0]', '(1206.0, 5501.0]']
        },
        "Age_of_Applicant": {
            'High Risk': ['(17.999, 25.0]', '(25.0, 30.0]'],
            'Medium Risk': ['(30.0, 35.0]', '(35.0, 41.0]'],
            'Low Risk': ['(41.0, 49.0]', '(49.0, 87.0]']
        },
        "Bureau_Score": {
            'High Risk': ['(646.999, 859.0]', '(859.0, 912.0]'],
            'Medium Risk': ['(912.0, 942.0]'],
            'Low Risk': ['(942.0, 972.0]', '(972.0, 1001.0]']
        }
    }

# coarse classing
    coarse_class_results = {}
    for var, mapping in coarse_bin_dicts.items():
        try:
            print(f"\n[TRACE] Coarse classing: {var}")
            fine_summary = fine_class_results[var]
            coarse_summary = coarse_class_variable(fine_summary, mapping)
            coarse_class_results[var] = coarse_summary

            bin_to_label = {str(b): label for label, bins in mapping.items() for b in bins}
            dev_df[f"{var}_bin"] = pd.cut(dev_df[var], bins=fine_summary['Bin'].cat.categories, include_lowest=True)
            dev_df[f"{var}_coarse"] = dev_df[f"{var}_bin"].astype(str).map(bin_to_label)

        except Exception as e:
            print(f"[ERROR] Issue with {var}: {e}")

# Calculate WoE and IV
    print("\n=== Weight of Evidence (WoE) & Information Value (IV) ===")
    woe_iv_results = {}
    for var in coarse_bin_dicts.keys():
        coarse_col = f"{var}_coarse"
        woe_df, iv = calculate_woe_iv(dev_df, coarse_col, target='GB_Flag')
        print(f"\nWoE table for {var}:\n", woe_df)
        print(f"IV for {var}: {iv:.4f}")
        woe_iv_results[var] = {
            "WOE": woe_df["WoE"].to_dict(),
            "IV": iv
        }

# Thereshhold Select variables with IV >= 0.1
    selected_vars = select_variables_by_iv(woe_iv_results, threshold=0.1)
    print(f"\nSelected variables for modeling (IV >= 0.1): {selected_vars}")

# WoE encoding
    for var in selected_vars:
        coarse_col = f"{var}_coarse"
        woe_col = f"{var}_woe"
        dev_df[woe_col] = dev_df[coarse_col].map(woe_iv_results[var]["WOE"])

#Final modeling dataset
    X_dev = dev_df[[f"{v}_woe" for v in selected_vars]].copy()
    y_dev = dev_df["GB_Flag"].map({"Good": 1, "Bad": 0})
    print("\n Final modeling dataset preview:")
    print(X_dev.head())
    print("\n Target variable distribution:")
    print(y_dev.value_counts())
    
    
# OLS Linear Regression
    import statsmodels.api as sm

    # Drop rows with NaNs
    X_dev_clean = X_dev.dropna()
    y_dev_clean = y_dev.loc[X_dev_clean.index]  # Align target with cleaned features

# Add intercept
    X_dev_with_const = sm.add_constant(X_dev_clean)

    ols_model = sm.OLS(y_dev_clean, X_dev_with_const).fit()

# Print summary
    print("\n=== OLS Linear Regression Summary ===")
    print(ols_model.summary())

    
# Logistic Regression

# Drop rows with NaNs
    X_dev_clean = X_dev.dropna()
    y_dev_clean = y_dev.loc[X_dev_clean.index]

# Add intercept
    X_dev_with_const = sm.add_constant(X_dev_clean)

    logit_model = sm.Logit(y_dev_clean, X_dev_with_const).fit()

# logot sumamry
    print("\n=== Logistic Regression Summary ===")
    print(logit_model.summary())
    
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

# Predict probabilities 
    y_probs = logit_model.predict(X_dev_with_const)

# calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_dev_clean, y_probs)
    roc_auc = auc(fpr, tpr)

# Plot for ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
