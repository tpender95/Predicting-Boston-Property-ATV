#  Teddy Pender's Captsone Submission

To begin with the prompt -  "The city of Boston conducts an annual tax assessment of all the properties in the Boston jurisdiction. As the city has to send officers to almost 200,000 properties every year. What they would like to know is whether automation can help them assess properties at a cheaper and at a faster pace a human might. The city of Boston would also be interested in possibly predicting growth in certain areas or buildings."

There are numerous things that came to my mind upon reading the prompt, they are: 
* Predicting the assessed tax value of a property through a regression.
* Time Series Analysis of the median assessed tax value of similar properties based on clustering techniques.

Data and information was was taken from the [City of Boston Assessing Department](https://data.boston.gov/dataset/property-assessment).

# EDA
Initially, I thought to seek out which features had the most influence on the target variables AV_TOTAL. 
Apart from other target variables the most highly correlated features were:
* LIVING_AREA = 0.77
* GROSS_AREA = 0.76

Before I began any feature engineering I decided to look at the target variables. Specifically I have included the visualizations for AV_TOTAL. There is a wide range of prices in all the training data frames and for this reason I decided to take the properties with an AV_TOTAL of 0 and add 1 to them, and then took the log transform of the entire column to see how the target variable was distributed. I felt that this was appropriate because firstly because log(1) = 1 and wouldn't offset the other data either to give as a somewhat normal distribution of the target variable. Predicting using the log transformation can is helpful be used to make highly skewed distributions less skewed. This can be valuable to both making patterns in the data more interpretable and also for helping to meet the assumptions of inferential statistics. 

The target variable's distribution for all years is shown below:
<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/logtransavtot.png" width="700" title="AV_TOTAL Distribution For All Years">
</p>

# Feature Engineering
To begin the feature engineering I began by looking at certain composing features. After which I decided that such polynomial features would likely just result in multicollinearity and decided this was not the best approach. I began with filtering the data frame by taking out any target variables that were greater than the 95th percentile. I took time to research and learn about certain features to better understand their importance and also whether they were truly numerical or categorical in nature. The most obvious one perhaps is a property's zip-code. Although this is numerical in value, it has no inherent numerical nature in the feature matrix. For the many other variables like zip-code I added them to a new data frame that contained exclusively categorical variables and then I applied the pandas operation of getting dummies for all the categorical variables and removed any unnecessary categorical variables like mailing address. Similarly, I also took all the numerical variables into a data frame alone and then when the two data frames had been appropriately formatted I concatenated the two data frames back together, because I haven't changed the rows order, I have assumed that the rows still correspond to the original property ID. I then applied the same methodology to the testing data frame for that given year also. Finally I dropped any reamining columns that the year's corresponding training and testing data frames did not have in common.

Another aspect I looked at was principal component analysis (PCA). PCA allowed me to reduce dimensionality of my data frames and therefore I now hand a much simpler model to run and take insights from. PCA also allows me to make the assumption that all of my principal components are linearly independent, this is a naive assumption that is often overlooked.

# Modeling
## Regression
Once the formatting of the data frames was complete I conducted a train-test-split with a test size of 85% of the original data frame, of original size 120,000, and then used the standard scaler function from sklearn's preprocessing library, fit it to the entire original data frame and then transformed the X_train and X_test data frames. From that point I used a Ridge regression in a grid-searching function called gsoptimizer that optimized for the negative mean squared error. I also ran the same model and optimized for the scorer R^2 and achieved a best score of 0.96. Once the model has run the function then predicts the true AV_TOTAL for the corresponding year's data frame and saves that to a .csv file.
<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/av_2018_truevpred.png" width="700" title="Ridge Regression of ATV for 2018, R^2 = 0.96">
</p>

I decided to use a ridge regression because a ridge regression is a regression technique that is optimized for prediction, rather than inference. Unlike a normal regression which gives you some unbiased regression coefficients which are just the maximum likelihood estimates that are observed in the given data set a ridge regression allows you to regularize the coefficients. What this means is that the coefficients estimated by the regression are pushed towards 0, to make them work better on a testing data set, this just means we're optimizing for prediction. Overall this lets use a somewhat complex model and while avoiding much over-fitting at the same time.

## Time Series & Clustering
Initially, I thought it would be possible to cluster the original data frame into groups given. To do so I used a k-means method of clustering. First, I looked to find an optimum number of clusters and this would just be the number of clusters where the inertia of the cluster, or the sum of squared distances, change was changing very little (we're looking for the "elbow of the graph"). That is apparent when $k=2$. This graph is shown below.

<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/elbow_kmeans.png" width="700" title="Optimized k">
</p>

However doing this results in clusters that are that are extremely unbalanced - I had it that there existed one cluster of ~119,000 properties in one cluster and ~100 in another. To combat this I chose a higher number of clusters and picked the top four clusters, for this I used k=15. I had four classes balanced at 59966, 25485, 21874 and 4712. It was these clusters I wanted to track the assessed tax value of. What we see when we look at the standard scaled price change over time is something very similar. It is apparent that after the financial crisis of 2008 housing prices rebounded in 2009 and 2010 and have since been steadily increasing. This can be seen below.

<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/cluster_avt_time_series.png" width="700" title="Standardized Assessed Tax Valuation Over Time">
</p>

I initially looked to see if there was any correlation with the previous year's median value with the current to attempt a time-lag model, but there was no significant correlations. I carried out a Random Forest regressor on a feature matrix which contained the rolling mean of two time steps of the true value - the models performance can be seen in the graph below.

<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/rf_price_prediction.png" width="700" title="Random Forest Regressor for Time Series Analysis of ATV">
</p>

However, I do believe that even though I have used the median house price change over the time period the data set lacks more sequential time series data to use more complex models like an LSTM. Perhaps there is something else to do here with this data but I am not aware of it. The most robust model I could truly thing about building is using the historical mean, the historical median for added robustness or even a random walk to forecast the last observation out. One could possibly view this problem as simply dealing with a sparse matrix with 364 missing values in-between each data point. We could seek to imput data into these rows however, that is making up data and not a method of best practice.

# Conclusion

