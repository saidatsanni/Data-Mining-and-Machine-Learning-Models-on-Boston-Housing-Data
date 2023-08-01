# Machine Learning Algorithms for Boston Housing Data

**Goal**

The goal of this study is to conduct a comprehensive analysis of Boston Housing Data aimed at identifying the model that provides the best prediction of median housing prices. Machine Learning methods such as Regression Models, Variable Selection, Regression Trees, Bagging, Random Forest, Boosted Regression Trees, Generalized Additive Models, and Neural Networks are considered. The in-sample and out-of-sample prediction accuracy and performance of these models are explored.

**Data**

The Boston Housing Data set which is available in R is used in this project.


The Boston dataset is a famous multivariate data set introduced in 1978 by Harrison, D. and Rubinfeld, D.L in their paper titled “Hedonic prices and the demand for clean air”. The data set includes 14 different quantitative variables. MEDV is the response variable which represents the median value of owner-occupied homes in $1000’s. The rest of the variables are predictive variables. The entire data set consists of 506 observations and 14 variables for predicting the prices of houses using the given features. 


**Statistical Techniques**

* Linear Regression Analysis
* Variable Selection (Best subset, forward selection, backward selectiom, Stepwise, and Lasso variable selection methods)
* Cross Validation
* Regression Trees
* Bagging
* Random Forest
* Boosted Regression Trees
* Generalized Additive Model (GAM)
* Neural Networks
**Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included**


**Major Findings and Conclusion**
[Model Comparison]()

Note that a total of 1,000 trees is considered for Bagging, Random Forest, and Boosting. 

Table 1 reveals that the trees perform better both in-sample and out-of-sample compared to the linear regression model. Meanwhile, the best regression tree which is a pruned tree of size 10, performs worse than the other tree models. This is because the other procedures are based on the averages of multiple trees on bootstrap samples (Bagging and Bootstrap) and sequential learning (Boosting). 

Meanwhile, the advanced tree methods: Bagging, Boosting, and Random Forest perform better than the GAM model for both in-sample and out-of-sample. However, the GAM model considers the non-linear relationship of the predictor variables with the response variable. 

Overall, the neural network has the best prediction performance. This method has the best out-of-sample performance compared to all the models considered. In conclusion, the neural network does a much better job at predicting compared to linear regression, tree models, and generalized additive model.
