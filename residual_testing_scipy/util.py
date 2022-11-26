import pandas as pd
import numpy as np

# Read in the excel file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   Y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_excel_data(infilename):
    df = pd.read_excel(infilename, index_col=0)
    n, d = df.values.shape
    d = d - 1  # The price column doesn't count
    X = df.values[:, :-1]
    labels = df.columns[:-1]
    Y = df.values[:, -1]
    X = np.hstack([np.ones((n, 1)), X])
    return X, Y, labels


# Make it pretty
def format_prediction(B, labels):
    str = f"predicted price = ${B[0]:,.2f} + "
    d = len(labels)
    for i in range(d):
        b = B[i + 1]
        label = labels[i]
        str += f"(${b:,.2f} x {label})"
        if i < d - 1:
            str += " + "
    return str


# Return the R2 score for coefficients B
# Given inputs X and outputs Y
def score(B, X, Y):
    mean_y = Y.mean()
    print(f"Mean house price is ${mean_y:,.2f}")
    residual_for_mean = Y - mean_y
    error_squared_for_mean = residual_for_mean @ residual_for_mean

    prediction = X @ B
    residual_for_lp = Y - prediction
    error_squared_for_lp = residual_for_lp @ residual_for_lp

    return 1 - error_squared_for_lp / error_squared_for_mean
