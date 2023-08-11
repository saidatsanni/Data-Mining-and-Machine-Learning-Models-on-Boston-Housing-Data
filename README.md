<meta name="google-site-verification" content="wZhvU0gEvoLqPgvTUkTG15T_KDHPgVFUQgp0rPffUUk" />

# Machine Learning Algorithms for Boston Housing Data

**Goal**

Comprehensive analysis of the Boston Housing Data aimed at identifying the model that provides the best prediction of median housing prices. Machine Learning models such as Regression Models, Variable Selection, Regression Trees, Bagging, Random Forest, Boosted Regression Trees, Generalized Additive Models, and Neural Networks are considered. The in-sample and out-of-sample prediction accuracy and performance of these models are explored.

**Data**

The Boston Housing Data set which is available in R is used in this project.


The Boston dataset is a famous multivariate data set introduced in 1978 by Harrison, D. and Rubinfeld, D.L in their paper titled “Hedonic prices and the demand for clean air”. MEDV is the response variable which represents the median value of owner-occupied homes in $1000’s. The data set consists of 506 observations and 14 predictor variables for predicting house prices. 


**Statistical Techniques**

* Linear Regression Analysis
* Variable and Model Selection (Best subset, forward selection, backward selection, Stepwise, and Lasso variable selection methods)
* Cross Validation
* Regression Trees
* Bagging
* Random Forest
* Boosted Regression Trees
* Generalized Additive Model (GAM)
* Neural Networks
* Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included**


**Major Findings and Conclusion**

![image](https://github.com/saidatsanni/Machine-Learning-Models-on-Boston-Housing-Data/assets/139437600/e864fbd5-52a0-47c6-920b-4119b0a0ed6e)

**Note that a total of 1,000 trees is considered for Bagging, Random Forest, and Boosting.**
**The data set is split into 70% Training data and 30% Testing data**

* The tree models provide better in-sample and out-of-sample prediction performance than the linear regression model.
* The best regression tree shown in the table is a pruned tree of size 10. This performs worse than the other tree models. This is likely because the other procedures are based on the averages of multiple trees on bootstrap samples (Bagging and Random Forest) and sequential learning (Boosting). 
* The advanced tree methods: Bagging, Boosting, and Random Forest perform better than the GAM model for both in-sample and out-of-sample.
* Overall, the neural network has the best prediction performance compared to all the models that are considered.


**Codes**

The codes can be found here: [Codes](https://github.com/saidatsanni/Machine-Learning-Models-on-Boston-Housing-Data/blob/0b304a99c9f387ad17593c0754721ffb939d45b0/Main/Machine%20Learning%20on%20Boston%20Housing%20Data.R)
