import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)

infilename = sys.argv[1]

df = pd.read_csv(infilename, index_col="property_id")
Y = df.values[:, -1]
print("Making new features...")
# As per the graph between age_of_roof doesn't have much significance
# We are combining two columns lot_width and lot_depth to create a new feature lot_size
# Creating a new feature is_close_to_school instead of miles_to_school
df["lot_size"] = df["lot_width"] * df["lot_depth"]
df["is_close_to_school"] = [1 if skl < 2.0 else 0 for skl in df["miles_to_school"]]
df["1"] = 1
X = pd.concat(
    [df["1"], df["sqft_hvac"], df["lot_size"], df["is_close_to_school"]], axis=1
)
labels_col = list(X.columns[1:])
print(f"Using only the useful ones: {labels_col}")
regr = LinearRegression(fit_intercept=False)
regr.fit(X, Y)
print(f"R2 = {regr.score(X,Y):.5f}")
print("*** Predication ***")
print(
    f"Price = ${round(regr.coef_[0],2):,} + (sqft x ${round(regr.coef_[1],2):,})+ (lot_size x ${round(regr.coef_[2],2):,})"
)
print(
    f"\t Less than 2 miles from a school? You get ${round(regr.coef_[3],2):,} added to the price!"
)