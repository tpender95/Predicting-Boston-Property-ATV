#  Teddy Pender's Captsone Submission

To begin with the prompt -  "The city of Boston conducts an annual tax assessment of all the properties in the Boston jurisdiction. As the city has to send officers to almost 200,000 properties every year. What they would like to know is whether automation can help them assess properties at a cheaper and at a faster pace a human might. The city of Boston would also be interested in possibly predicting growth in certain areas or buildings."

There are numerous things that came to my mind upon reading the prompt, they are: 
* Predicting the assessed tax value of a property through a regression.
* Time Series Analysis of the median assessed tax value of similar properties based on clustering techniques.

# EDA
Initially, I thought to seek out which features had the most influence on the target variables AV_TOTAL. 
Apart from other target variables the most highly correlated features were:
* LIVING_AREA = 0.77
* GROSS_AREA = 0.76

Before I began any feature engineering I decided to look at the target variables. Specifically I have included the visualizations for AV_TOTAL. There is a wide range of prices in all the training data frames and for this reason I decided to take the properties with an AV_TOTAL of 0 and add 1 to them, and then took the log transform of the entire column to see how the target variable was distributed. I felt that this was appropriate because firstly because log(1) = 1 and wouldn't offset the other data either to give as a somewhat normal distribution of the target variable. Predicting using the log transformation can is helpful be used to make highly skewed distributions less skewed. This can be valuable to both making patterns in the data more interpretable and also for helping to meet the assumptions of inferential statistics. 

The target variable's distribution for all years is shown below:
<p align="center">
  <img src="https://github.com/tpender95/Predicting-Boston-Property-ATV/blob/master/pics/logtransavtot.png" width="350" title="AV_TOTAL Distribution For All Years">
</p>

