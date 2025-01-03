import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import numpy as np

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
    # Filter valid data
    data = data[(data['TotalPremium'] > 0) & (data['TotalClaims'] >= 0)].copy()
    data.dropna(subset=['PostalCode'], inplace=True)
    
    # Calculate Profit Margin
    data['ProfitMargin'] = (data['TotalPremium'] - data['TotalClaims']) / data['TotalPremium']
    data['ProfitMargin'] = data['ProfitMargin'].fillna(0)  # Avoid chained assignment warning
    
    # Group by ZIP codes
    zipcode_groups = [data[data['PostalCode'] == z]['ProfitMargin'] for z in data['PostalCode'].unique()]
    valid_groups = [group for group in zipcode_groups if len(group) > 1]
    
    # Check for sufficient valid groups
    if len(valid_groups) < 2:
        return {
            "Test": "ANOVA",
            "Null Hypothesis": "No significant margin differences between ZIP codes",
            "F-Statistic": np.nan,
            "p-Value": np.nan,
            "Reject Null": False
        }
    
    # Perform ANOVA
    result = f_oneway(*valid_groups)
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
    Run all hypothesis tests and visualize results.
    """
    results = []
    results.append(test_risk_across_provinces(data))
    results.append(test_risk_between_zipcodes(data))
    results.append(test_margin_difference_between_zipcodes(data))
    results.append(test_risk_difference_gender(data))
    
    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results)
    
    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Bar plot for p-Values
    ax[0].barh(results_df['Test'], results_df['p-Value'], color='skyblue')
    ax[0].set_xlabel('p-Value')
    ax[0].set_title('p-Value for Each Test')
    
    # Bar plot for Null Hypothesis Rejection
    ax[1].barh(results_df['Test'], results_df['Reject Null'].astype(int), color='lightcoral')
    ax[1].set_xlabel('Reject Null (1 = Yes, 0 = No)')
    ax[1].set_title('Null Hypothesis Rejection for Each Test')
    
    plt.tight_layout()
    plt.show()
    
    return results_df
