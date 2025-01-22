import numpy as np
import pandas as pd
from scipy.stats import shapiro, kstest, normaltest
import matplotlib.pyplot as plt
import seaborn as sns

def check_normality(data, group_name):
    print(f"\nChecking normality for {group_name}:")
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(data)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")

    # Kolmogorov-Smirnov Test
    ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

    # D'Agostino and Pearson Test
    dag_stat, dag_p = normaltest(data)
    print(f"D'Agostino and Pearson Test: Statistic={dag_stat:.4f}, p-value={dag_p:.4f}")

    # Histogram and KDE plot
    sns.histplot(data, kde=True, stat="density", bins=15, label=group_name, color="blue")
    plt.title(f"Distribution of {group_name}")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f'{group_name} distribution.png')
    
if __name__ == '__main__':
    print ('p-value > 0.05 means possible Normal Distribution')
    group1 = np.random.normal(0, 1, 100)
    check_normality(group1, 'Normal Distribution')
    
    group2 = np.random.exponential(1, 100)
    check_normality(group2, 'Exponential Distribution')
    
