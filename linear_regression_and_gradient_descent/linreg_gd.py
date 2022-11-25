import numpy as np
import pandas as pd
import sys
import util
import matplotlib.pyplot as plt

# Check the command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Learning rate
t = 0.001

# Limit interations
max_steps = 1000

# Get the arg and read in the spreadsheet
infilename = sys.argv[1]
X, Y, labels = util.read_excel_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Get the mean and standard deviation for each column
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Don't mess with the first column (the 1s)
## Your code here
X = X[:,1:]

# Standardize X to be X' 
Xp = (X - X_mean[1:])/ X_std[1:]
Xp = np.append(np.ones([len(Xp),1]),Xp,axis=1)


# First guess for B is "all coefficents are zero"
B = np.zeros([6])

# Create a numpy array to record avg error for each step
errors = np.array([np.square(np.dot(Xp,B) - Y).mean()])
for i in range(max_steps):
 
    # Compute the gradient 
    gradient = np.dot(Xp.T,(np.dot(Xp,B) - Y))

    # Compute a new B (use `t`)
    B = B-np.dot(t,gradient)

    # Figure out the average squared error using the new B
    mse = np.square(np.dot(Xp,B) - Y).mean()

    # Store it in `errors``
    #errors = np.append(errors,mse,axis=0)
    errors = np.append(errors,mse)

    # Check to see if we have converged
    if round(errors[-1],2)-round(errors[-2],2)==0:
        break

print(f"Took {i} iterations to converge")

# "Unstandardize" the coefficients
B_new = B[1:]/X_std[1:]
B0 = np.subtract(B[0],np.dot(B[1:],np.divide(X_mean[1:],X_std[1:])))
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
B = np.append(B0,B_new)
X = np.append(np.ones([len(X),1]),X,axis=1)

# Show the result
print(util.format_prediction(B, labels))

# Get the R2 score
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# Draw a graph
fig1 = plt.figure(1, (4.5, 4.5))
axes = plt.axes()
axes.plot([iteration for iteration in range(errors.size)],errors)
axes.set(xlabel='Iterations',title='Convergence',xscale='log',yscale='log')
axes.set_ylabel('Mean squared error',labelpad=0)
## Your code ehre
fig1.savefig("err.png")
