# Post-Earthquake-Damage-Prediction
This model applies machine learning algorithm to a dataset after thoroughly processing and cleansing  the data, which contains previously damaged building due to earthquake and predict the damage a particular building might incur.

APPROACH:

This model applies machine learning algorithm to a kaggle dataset, which contains 2000000 rows of building data and its past damage degree it had incurred (from Grade1 -5). I performed a MULTI-CLASS Classification,  After thorough processing and cleaning the noisy and highly spread (Skewed) data,  I had to infer many insights from visualizing the data by correlation data and various countplots to get the main top 7 features from total of 156 features( after one hot encoding), and discarding the rest.
To this data, i applied random forest, xgboost and a deep ANN with 80596 parameters(nodes) , each with a cross-validated upto 10 times, to attain better accuracy(after optimizing) of 0.70876 , i.e. 70% Accuracy.  

· Output: 70% accuracy on validation set.
