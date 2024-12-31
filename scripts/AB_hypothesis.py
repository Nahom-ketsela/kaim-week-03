import pandas as pd
from scipy.stats import f_oneway, ttest_ind

def test_risk_across_provinces(self):
    """
    Test if there are significant risk differences (Total Claims) across provinces.
    Null Hypothesis: There are no risk differences across provinces.
    """
    province_groups = [self.data[self.data['Province'] == p]['TotalClaims'] 
                       for p in self.data['Province'].unique()]
    result = f_oneway(*province_groups)
    self.results.append({
        "Test": "ANOVA",
        "Null Hypothesis": "No risk differences across provinces",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    })

def test_risk_between_zipcodes(self):
    """
    Test if there are significant risk differences (Total Claims) between ZIP codes.
    Null Hypothesis: There are no risk differences between ZIP codes.
    """
    zipcode_groups = [self.data[self.data['PostalCode'] == z]['TotalClaims'] 
                      for z in self.data['PostalCode'].unique()]
    result = f_oneway(*zipcode_groups)
    self.results.append({
        "Test": "ANOVA",
        "Null Hypothesis": "No risk differences between ZIP codes",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    })

def test_margin_difference_between_zipcodes(self):
    """
    Test if there are significant margin differences (Profit Margin) between ZIP codes.
    Null Hypothesis: There are no significant margin differences between ZIP codes.
    """
    self.data['ProfitMargin'] = (self.data['TotalPremium'] - self.data['TotalClaims']) / self.data['TotalPremium']
    zipcode_groups = [self.data[self.data['PostalCode'] == z]['ProfitMargin'] 
                      for z in self.data['PostalCode'].unique()]
    result = f_oneway(*zipcode_groups)
    self.results.append({
        "Test": "ANOVA",
        "Null Hypothesis": "No significant margin differences between ZIP codes",
        "F-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    })

def test_risk_difference_gender(self):
    """
    Test if there are significant risk differences (Total Claims) between genders.
    Null Hypothesis: There are no significant risk differences between women and men.
    """
    male_group = self.data[self.data['Gender'] == 'Male']['TotalClaims']
    female_group = self.data[self.data['Gender'] == 'Female']['TotalClaims']
    result = ttest_ind(male_group, female_group, equal_var=False)
    self.results.append({
        "Test": "T-Test",
        "Null Hypothesis": "No significant risk differences between women and men",
        "T-Statistic": result.statistic,
        "p-Value": result.pvalue,
        "Reject Null": result.pvalue < 0.05
    })



def run_all_tests(self):
    """
    Run all hypothesis tests and save results.
    """
    self.test_risk_across_provinces()
    self.test_risk_between_zipcodes()
    self.test_margin_difference_between_zipcodes()
    self.test_risk_difference_gender()
