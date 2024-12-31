import pandas as pd
from scipy.stats import f_oneway, ttest_ind

def test_risk_across_provinces(data):
    """
    Test if there are significant risk differences (Total Claims) across provinces.
    Null Hypothesis: There are no risk differences across provinces.
    """
    province_groups = [data[data['Province'] == p]['TotalClaims'] for p in data['Province'].unique()]
    result = f_oneway(*province_groups)
    return {
        "Test": "ANOVA",
        "Null Hypothesis": "No risk differences across provinces",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    }

def test_risk_between_zipcodes(data):
    """
    Test if there are significant risk differences (Total Claims) between ZIP codes.
    Null Hypothesis: There are no risk differences between ZIP codes.
    """
    zipcode_groups = [data[data['PostalCode'] == z]['TotalClaims'] for z in data['PostalCode'].unique()]
    result = f_oneway(*zipcode_groups)
    return {
        "Test": "ANOVA",
        "Null Hypothesis": "No risk differences between ZIP codes",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    }

def test_margin_difference_between_zipcodes(data):
    """
    Test if there are significant margin differences (Profit Margin) between ZIP codes.
    Null Hypothesis: There are no significant margin differences between ZIP codes.
    """
    data['ProfitMargin'] = (data['TotalPremium'] - data['TotalClaims']) / data['TotalPremium']
    zipcode_groups = [data[data['PostalCode'] == z]['ProfitMargin'] for z in data['PostalCode'].unique()]
    result = f_oneway(*zipcode_groups)
    return {
        "Test": "ANOVA",
        "Null Hypothesis": "No significant margin differences between ZIP codes",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    }

def test_risk_difference_gender(data):
    """
    Test if there are significant risk differences (Total Claims) between genders.
    Null Hypothesis: There are no significant risk differences between women and men.
    """
    male_group = data[data['Gender'] == 'Male']['TotalClaims']
    female_group = data[data['Gender'] == 'Female']['TotalClaims']
    result = ttest_ind(male_group, female_group, equal_var=False)
    return {
        "Test": "T-Test",
        "Null Hypothesis": "No significant risk differences between women and men",
        "T-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    }

def run_all_tests(data):
    """
    Run all hypothesis tests and return results.
    """
    results = []
    results.append(test_risk_across_provinces(data))
    results.append(test_risk_between_zipcodes(data))
    results.append(test_margin_difference_between_zipcodes(data))
    results.append(test_risk_difference_gender(data))
    return results
