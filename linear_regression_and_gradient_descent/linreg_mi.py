import numpy as np
import pandas as pd
import sys
import util

# Check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Read in the argument
infilename = sys.argv[1]

# Read the spreadsheet
X, Y, labels = util.read_excel_data(infilename)

n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Find the coefficients for the linear regression
B = np.linalg.inv(X.T @ X) @ X.T @ Y

# Pretty print them
print(util.format_prediction(B, labels))

R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")
