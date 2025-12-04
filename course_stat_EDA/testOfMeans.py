# %% [markdown]
# #### Wilcoxon Signed rank sum Test
# For data that is not normally distributed
# 
# Example: before and after data (same individuals)

# %%
import scipy.stats as stats

# Sample data
pain_before = [7, 6, 8, 9, 5, 7, 8, 6, 7, 9, 6, 5, 4, 7, 6, 7, 8, 6, 5, 7]
pain_after = [5, 4, 6, 8, 3, 6, 7, 4, 6, 8, 4, 3, 2, 5, 4, 5, 7, 4, 3, 6]

# Perform Wilcoxon Signed Rank Sum Test
statistic, p_value = stats.wilcoxon(pain_before, pain_after)

# Output the results
print("Wilcoxon Signed Rank Sum Test")
print("----------------------------")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# %% [markdown]
# #### Example for paired t-Test (assumes normality):

# %%

# Sample data
blood_pressure_before = [140, 150, 155, 180, 144, 152, 187, 160, 150] 
blood_pressure_after = [130, 140, 145, 160, 130, 150, 173, 150, 140]  

# Perform paired t-test
statistic, p_value = stats.ttest_rel(blood_pressure_after, blood_pressure_before)

# Output the results
print("Paired t-test")
print("-------------")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# %% [markdown]
# #### Example for t-test between independent groups

# %%

# Sample data
group_a = [5, 6, 4, 6, 7, 8, 5, 7, 6, 7]
group_b = [7, 8, 6, 5, 6, 7, 8, 7, 6, 7]

# Perform independent t-test
statistic, p_value = stats.ttest_ind(group_a, group_b)

# Output the results
print("Independent t-test")
print("-------------")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# %% [markdown]
# #### Example for Non-parametric Comparison of Two Groups: Mann-Whitney Test

# %%

# Sample data
treatment_a = [3.5, 2.8, 4.2, 4.6, 2.9, 3.4, 4.1, 3.2, 3.8, 4.3, 3.7, 3.9, 4.2, 3.3, 3.6, 4.0, 4.1, 3.1, 3.2, 3.8, 4.0, 3.9, 3.7, 3.4, 3.5, 4.1, 4.3, 3.6, 4.0, 4.2]
treatment_b = [2.2, 3.9, 2.7, 3.5, 2.8, 3.1, 2.9, 2.7, 3.2, 2.6, 3.4, 3.0, 2.7, 3.3, 2.8, 3.0, 3.5, 2.9, 3.4, 2.8, 2.9, 2.6, 3.1, 3.0, 2.8, 3.5, 3.6, 3.3, 3.2, 2.7, 3.0, 2.8, 3.1, 3.2, 3.5, 4.0]

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(treatment_a, treatment_b)

# Output the results
print("Mann-Whitney U test")
print("-------------")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# %% [markdown]
# #### Example for ANOVA (parametric):
# Useful when indepedent variable is categorical (at least 3 categories), to compare multiple groups or to compare more than one independent variable.

# %%

# Sample data
drug_a = [10, 12, 8, 9, 11, 9, 11, 10, 12, 10, 9, 11, 10, 11, 8, 9, 12, 10, 9, 10, 12, 11, 10, 9, 11, 8, 10, 9, 12, 11]
drug_b = [13, 11, 14, 12, 15, 14, 12, 13, 11, 12, 15, 13, 14, 12, 13, 15, 12, 14, 13, 15, 11, 12, 14, 13, 15, 14, 12, 13, 11, 12]
drug_c = [9, 10, 11, 8, 10, 9, 11, 8, 9, 10, 11, 10, 8, 9, 11, 9, 10, 8, 10, 9, 10, 11, 9, 8, 10, 9, 11, 8, 10, 9]

# Perform one-way ANOVA
statistic, p_value = stats.f_oneway(drug_a, drug_b, drug_c)

# Output the results
print("One-Way ANOVA")
print("-------------")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

 

# %%



