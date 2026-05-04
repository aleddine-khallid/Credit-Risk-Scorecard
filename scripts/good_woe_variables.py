
def select_variables_by_iv(woe_iv_results, threshold=0.1):
    """
    Filters WOE/IV results dictionary and returns variable names with IV >= threshold.
    """
    selected_vars = []
    for var, summary in woe_iv_results.items():
        iv = summary.get("IV", 0)
        if iv >= threshold:
            selected_vars.append(var)
    return selected_vars