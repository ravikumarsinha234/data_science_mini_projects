## Analyze the residual 

Create a histogram of the residual as res hist.png. Use scipy’s kstest to confirm that the residual really resembles a normal distribution. The test returns a P-value; if the P-value is less than 0.05, you can
assume the residual is normally distributed. Now that you know it is a normal distribution, extend linreg scikit.py yet again to print your
confidence like this ”68% of the estimates done with this formula will be within $89.12 of the correct
price. 95% will be within $140.19 of the correct price.”

The output should look similar as mentioned below:

> python3 linreg_scikit.py properties.xlsx  

Read 519 rows, 5 features from ’properties.xlsx’.
predicted price = $32,362.85 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +  
($59,195.07 x bedrooms) + ($9,599.24 x bathrooms) +  
($-17,421.84 x miles_to_school)  
Kolmogorov-Smirnov: P-value = 4.154181404788638e-129
The residual follows a normal distribution.
68% of predictions with this formula will be within $91,849.54 of the actual price.
95% of predictions with this formula will be within $183,699.08 of the actual price.
