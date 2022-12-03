import numpy as np
import pandas as pd
from scipy.stats import chi2

# Read in the data
mice_df = pd.read_csv("mice.csv")

# Figure out the possible gene types
gene_types = list(mice_df.gene_type.unique())
print(f"Possible gene types:{gene_types}")

# create a contingency table for gene_type column with pandas crosstab
contingency_table = pd.crosstab(
    index=mice_df.gene_type, columns=mice_df.has_cancer, margins=True
)
contingency_table.columns = ["No Cancer", "Has Cancer", "Total"]
contingency_table.index = gene_types + ["Total"]
print(f"\n**** Contingency_Table ****")
print(contingency_table)

# create conditional proportions table
proportions_table = (
    contingency_table.iloc[0:4, 0:2].div(contingency_table["Total"], axis=0) * 100
).round(2)
proportions_table["Proportion"] = (
    contingency_table["Total"] / contingency_table["Total"][-1] * 100
).round(2)
proportions_table_print = proportions_table.astype(str) + "%"
proportions_table_print.iloc[3:4, 2:3] = ""
print("\n**** Proportions_Table ****")
print(proportions_table_print)

# create expected values table from the conditional proportions table if gene and cancer are independent
expected_values_table = pd.DataFrame()
expected_values_table["False"] = (
    contingency_table["Total"] * float(proportions_table.iloc[-1, 0:1] / 100)
).round(2)
expected_values_table["True"] = (
    contingency_table["Total"] * float(proportions_table.iloc[-1, 1:2] / 100)
).round(2)
expected_values_table["Total"] = (expected_values_table.sum(axis=1)).round(2)
expected_values_table.iloc[-1:, 0:2] = (
    proportions_table.iloc[-1:, 0:2].astype(str) + "%"
)
expected_values_table.iloc[3:4, 2:3] = ""
expected_values_table.columns = ["No Cancer", "Has Cancer", "Total"]
print("\n**** Expected_Table ****")
print(expected_values_table)
observed_matrix = np.matrix(contingency_table.iloc[0:3, 0:2])
expected_matrix = np.matrix(expected_values_table.iloc[0:3, 0:2])
# calculate chi-square value
numerator = np.square(observed_matrix - expected_matrix)
x2 = np.sum(numerator / expected_matrix)
print(f"\nchi2 value:{x2}")
# calculate degrees of freedom
dof = (len(contingency_table.index) - 2) * (len(contingency_table.columns) - 2)
print(f"degrees of freedom: {dof}")
# calculate p-value
p_value = 1 - chi2.cdf(x2, dof)
print(f"p-value: {p_value}")
# print the proclamation as per p-value
if p_value < 0.05:
    print(
        "It seems very, very unlikely that we would have seen these numbers if the gene and cancer were independent."
    )
else:
    print(
        "It is very likely that we would have seen these numbers if the gene and cancer are dependent."
    )
