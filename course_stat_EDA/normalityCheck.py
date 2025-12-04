# %% [markdown]
# # Normality Check
# 

# %% [markdown]
# #### Q-Q Plot

# %%
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate random data from a normal distribution
np.random.seed(123)
data = np.random.normal(loc=0, scale=1, size=100)

# Create the QQ-plot
fig, ax = plt.subplots()
sm.qqplot(data, line='s', ax=ax)

# Customize the plot
ax.set_title('QQ-Plot')
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')

# Display the plot
plt.show()

# %% [markdown]
# Another exampe: It is manually caclualte qqplot.

# %%
import scipy.stats as stats

# Generate a sample dataset
np.random.seed(0)
sample_data = np.random.normal(loc=0, scale=1, size=1000)

# Generate theoretical quantiles from a normal distribution
theoretical_quantiles = np.linspace(-3, 3, 100)

# Calculate the quantiles of the sample data
sample_quantiles = np.percentile(sample_data, np.linspace(0, 100, 100))

# Create the QQ plot
plt.figure(figsize=(6, 6))
plt.scatter(theoretical_quantiles, sample_quantiles)
plt.plot([-3, 3], [-3, 3], color='red', linestyle='--')  # Add a reference line
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('QQ Plot')
plt.grid(True)
plt.show()

# %% [markdown]
# #### P-P plots:

# %%

# Generate random data from a normal distribution
np.random.seed(123)
data = np.random.normal(loc=0, scale=1, size=100)

# Fit the data to a normal distribution
mu, std = np.mean(data), np.std(data)
norm_dist = np.random.normal(loc=mu, scale=std, size=len(data))

# Create the P-P plot
fig, ax = plt.subplots()
sm.ProbPlot(data).ppplot(line='45', ax=ax)

# Customize the plot
ax.set_title('P-P Plot')
ax.set_xlabel('Theoretical Cumulative Probabilities')
ax.set_ylabel('Sample Cumulative Probabilities')

# Display the plot
plt.show()

# %% [markdown]
# #### Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling

# %% [markdown]
# In this example, we first generate a random dataset from a normal distribution using numpy. Then, we use the respective functions from the scipy.stats module to perform the tests.
# 
# The Shapiro-Wilk test is performed using stats.shapiro(). The Kolmogorov-Smirnov test is performed using stats.kstest() with the distribution set to 'norm' for the normal distribution. The Anderson-Darling test is performed using stats.anderson() with dist='norm' to specify the normal distribution.
# 
# The code outputs the test statistics, p-values, critical values (for Anderson-Darling test), and significance levels.
# 
# Note: Make sure to have the numpy and scipy libraries installed in your Python environment before running this code.

# %%

# Generate a random dataset
np.random.seed(123)
data = np.random.normal(loc=0, scale=1, size=100)

# Shapiro-Wilk test
shapiro_test = stats.shapiro(data)
print("Shapiro-Wilk Test:")
print("Test Statistic:", shapiro_test.statistic)
print("p-value:", shapiro_test.pvalue)

# Kolmogorov-Smirnov test
ks_test = stats.kstest(data, 'norm')
print("\nKolmogorov-Smirnov Test:")
print("Test Statistic:", ks_test.statistic)
print("p-value:", ks_test.pvalue)

# Anderson-Darling test
anderson_test = stats.anderson(data, dist='norm')
print("\nAnderson-Darling Test:")
print("Test Statistic:", anderson_test.statistic)
print("Critical Values:", anderson_test.critical_values)
print("Significance Levels:", anderson_test.significance_level)

# %% [markdown]
# **Question: how do you interpreate this data - related to normalities?**

# %%

# Given volumes
volumes = np.array([255.2, 248.5, 251.9, 257.6, 243.8, 249.1, 253.4, 246.7, 250.8, 242.3, 258.9, 244.6, 247.2, 251.3, 254.7])
mean_volume = np.mean(volumes)
target_volume = 250

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(volumes, target_volume)

# Print results
print("Mean Volume:", mean_volume)
print("Target Volume:", target_volume)
print("t-statistic:", t_stat)
print("p-value:", p_value)


# %% [markdown]
# **Test for Normality**

# %%
# Given volumes
volumes = np.array([255.2, 248.5, 251.9, 257.6, 243.8, 249.1, 253.4, 246.7, 250.8, 242.3, 258.9, 244.6, 247.2, 251.3, 254.7])

# Shapiro-Wilk test for normality
shapiro_test = stats.shapiro(volumes)

# Print test statistic and p-value
print("Shapiro-Wilk Test:")
print("Test Statistic:", shapiro_test.statistic)
print("p-value:", shapiro_test.pvalue)


