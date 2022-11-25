# Programs on Linear Regression by 3 different ways

You are going to create three python programs:  
• linreg mi.py uses matrix inversion to come up with the formula.  
• linreg sckit.py uses scikit-learn to find the formula.  
• linreg gd.py uses gradient to converge upon the formula.  


All three take the filename of the spreadsheet as an argument:  
    > python3 linreg_mi.py properties.xlsx  
    

The program will read the excel spreadsheet that has the features of houses and the price they sold for:  
All three will find the hyperplane that minimizes the L2 error for those 519 data points.  
Each program will output those coefficients as a formula for predicting house prices:  

predicted price = $32,362.85 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +
($59,195.07 x bedrooms) + ($9,599.24 x bathrooms) +
($-17,421.84 x miles_to_school)}


## linreg gd.py

In this, you will be using gradient descent to minimize the squared error. The result should be very
nearly the same as linreg mi.py.  
The features in properties.xlsx are on very different scales (2 bathrooms vs 50,000 square foot
yards). As a result, converging would take a very, very long time if you don’t first standardize the
features.  
The first step is to find the mean and the standard deviation of each column of X. Use those to
make each feature have a mean of 0 and a standard deviation of 1.  
Then start with a guess of zero for all the coefficients. Do the following many times:  
• Calculate the gradient  
• Update your guess. (Multiply the gradient by -0.001 and add to the last guess.)  
• Compute and record the new mean squared error  
When the gradient gets small (and thus the changes to the coefficients gets small), stop. It should
take a few hundred iterations.  
The coefficients that you have calculated are for standardized inputs. Using the means and standard
deviations you computed early, adjust them to use unstandardized data. (The math for this is in
the next section.  

## Sample Output

We expect the following output:  
> python3 linreg_mi.py properties.xlsx  
Read 519 rows, 5 features from ’properties.xlsx’.  
predicted price = $32,362.85 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +  
($59,195.07 x bedrooms) + ($9,599.24 x bathrooms) + ($-17,421.84 x miles_to_school)  
R2 = 0.875699  


> python3 linreg_scikit.py properties.xlsx  
Read 519 rows, 5 features from ’properties.xlsx’.  
predicted price = $32,362.85 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +  
($59,195.07 x bedrooms) + ($9,599.24 x bathrooms) + ($-17,421.84 x miles_to_school)  
R2 = 0.875699  


> python3 linreg_gd.py properties.xlsx  
Read 519 rows, 5 features from ’properties.xlsx’.  
Took 352 iterations to converge  
predicted price = $32,362.82 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +  
($59,196.55 x bedrooms) + ($9,598.99 x bathrooms) + ($-17,421.85 x miles_to_school)  
R2 = 0.875699  
