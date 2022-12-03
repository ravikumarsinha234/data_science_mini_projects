# How to perform the testing?

Some times we will look at two categorical variables and try to figure out if they are related. Does
knowing that the mouse has a particular gene tell us anything about the probability that it will
get cancer?  
We are given a csv with the results of this sort of experiment called mice.csv. Write a program
check mice.py that does the analysis. 
For example, we should start out with a contingency table:  

![image](https://user-images.githubusercontent.com/47293331/205447449-6e483a94-3699-4ecf-918e-52feb54776fc.png)  

  
Note the degrees of freedom. (It is 2.)  
And do a p-test:  
p = 2.853273173286652 × 10−14  
And then give proclamation: ”It seems very, very unlikely that we would have seen these numbers
if the gene and cancer were independent.”  
