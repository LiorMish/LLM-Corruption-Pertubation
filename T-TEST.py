import pandas as pd
import numpy as np
import scipy.stats as stats

# Load the CSV files
file_path1 = "C:/Users/user/Desktop/OAI/experiment_results_pinguin.csv"  # Adjust the path if needed
file_path2 = "C:/Users/user/Desktop/OAI/experiment_results_la_la.csv"  # Adjust the path if needed
file_path3 = "C:/Users/user/Desktop/OAI/experiment_results_dedicated.csv"
# file_path4 = "C:/Users/user/Desktop/OAI/experiment_results_shimdura.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
# df4 = pd.read_csv(file_path4)

# Merge the two dataframes
df = pd.concat([df1, df2, df3], ignore_index=True)

# Extract perplexity values
clean_perplexity = df["Clean Perplexity"].dropna()
attacked_perplexity = df["Attacked Perplexity"].dropna()

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(clean_perplexity, attacked_perplexity, equal_var=False)

# Display results 4 digits after the decimal point
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
