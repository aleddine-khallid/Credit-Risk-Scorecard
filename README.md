#  Credit Risk Scorecard Modeling Pipeline

##  Project Overview
This project implements a full end-to-end **Credit Scoring Pipeline** designed to predict the likelihood of a loan applicant defaulting (**GB Flag**). Developed as a summer research project, it follows industry-standard financial modeling techniques to transform raw bank application data into a predictive model capable of distinguishing between "Good" and "Bad" credit risks.

---

##  Feature Engineering
The core of the pipeline focuses on robust statistical transformations to ensure model stability and interpretability:

*   **Fine Classing:** Initial granular binning of continuous variables to capture non-linear trends.
*   **Coarse Classing:** Strategic grouping of bins based on similar risk profiles to ensure monotonicity.
*   **Statistical Selection:** Identification of the most predictive variables using **Information Value (IV)** with a threshold of $IV \ge 0.1$.
*   **WoE Encoding:** Transformation of categorical/binned data into **Weight of Evidence** statistical weights.

---

##  Model Comparison & Performance
The project evaluates different modeling approaches to balance performance and explainability.

Models :

**OLS Linear Regression** 
 **Logistic Regression** 
 **XGBoost** 

> **Validation:** Models are evaluated using **ROC Curves** and **AUC (Area Under the Curve)** scores.

---

## Tech Stack
*   **Language:** Python
*   **Data Analysis:** `pandas`, `numpy`
*   **Statistics & Modeling:** `statsmodels`, `scikit-learn`
*   **Visualization:** `matplotlib`

---

