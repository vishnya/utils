# PLOT CORRELATIONS


def plot_chi2_heatmap(df, columns_to_compare):
    # columns_to_compare e.g. df.columns.values
    from scipy.stats import chi2_contingency
    import pandas as pd
    import numpy as np
    import seaborn as sns

    factors_paired = [(i, j) for i in columns_to_compare for j in columns_to_compare]

    chi2, p_values = [], []

    for f in factors_paired:
        if f[0] != f[1]:
            chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))
            chi2.append(chitest[0])
            p_values.append(chitest[1])
        else:
            chi2.append(0)
            p_values.append(0)

    chi2 = np.array(chi2).reshape((len(columns_to_compare), len(columns_to_compare)))  # shape it as a matrix
    chi2 = pd.DataFrame(chi2, index=columns_to_compare, columns=columns_to_compare)
    sns.heatmap(chi2)
