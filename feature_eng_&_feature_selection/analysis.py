import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# Deal with command-line
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)
infilename = sys.argv[1]

# Read in the basic data frame
df = pd.read_csv(infilename, index_col="property_id")
X_basic = df.values[:, :-1]
labels_basic = df.columns[:-1]
Y = df.values[:, -1]

# Expand to a 2-degree polynomials
poly = PolynomialFeatures(2)
poly_X_basic = poly.fit_transform(X_basic)

# Prepare for loop
residual = Y

# We always need the column of zeros to
# include the intercept
feature_indices = [0]
poly_feat_name = poly.get_feature_names_out(labels_basic)
print("First time though: using original price data as the residual")

while len(feature_indices) < 3:
    p_val_lst = []
    for i in range(1, len(poly_feat_name)):
        _, p_val = pearsonr(poly_X_basic[:, i], residual)
        print(f'\t"{poly_feat_name[i]}" vs residual: p-value={p_val}')
        p_val_lst.append(p_val)
        lowest_pval = np.argmin(p_val_lst)
    feature_indices.append(1 + lowest_pval)
    print(f"**** Fitting with {poly_feat_name[feature_indices]} ****")
    reg = LinearRegression().fit(poly_X_basic[:, feature_indices], Y)
    y_hat = reg.predict(poly_X_basic[:, feature_indices])
    print(f"R2 = {reg.score(poly_X_basic[:,feature_indices], Y)}")
    residual = Y - y_hat
    print(f"Residual is updated")

# Any relationship between the final residual and the unused variables?
print("Making scatter plot: age_of_roof vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 3], residual, marker="+")
fig.savefig("ResidualRoof.png")

print("Making a scatter plot: miles_from_school vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 4], residual, marker="+")
fig.savefig("ResidualMiles.png")