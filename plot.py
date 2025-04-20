import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import numpy as np

# Load the CSV file
file_path = "C:/Users/user/Desktop/OAI/experiment_results_pinguin.csv"  # Adjust the path if needed
file_path2 = "C:/Users/user/Desktop/OAI/experiment_results_la_la.csv"  # Adjust the path if needed
file_path3 = "C:/Users/user/Desktop/OAI/experiment_results_dedicated.csv"  # Adjust the path if needed
file_path4 = "C:/Users/user/Desktop/OAI/experiment_results_shimdura.csv"
df = pd.read_csv(file_path)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)

# Merge the two dataframes
df = pd.concat([df, df2, df3], ignore_index=True)

# Extract perplexity values
models = ["Clean RAG", "Attacked RAG"]
perplexities = [
    df["Clean Perplexity"].dropna(),
    df["Attacked Perplexity"].dropna(),

]

# Compute mean and standard error of the mean (SEM) for confidence intervals
means = np.array([p.mean() for p in perplexities])
errors = np.array([sem(p) for p in perplexities])  # Ensure it's a 1D NumPy array

# Create bar plot with error bars
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=models, y=means, capsize=0.2, hue=models, dodge=False, palette="coolwarm")

# Add error bars manually using Matplotlib
plt.errorbar(models, means, yerr=errors, fmt='none', capsize=5, color='black', elinewidth=1.5)

# Customize plot
plt.xlabel("Model Type")
plt.ylabel("Average Perplexity")
plt.title("Comparison of Perplexity")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend([],[], frameon=False)  # Remove legend if not needed

# Show plot
plt.show()
