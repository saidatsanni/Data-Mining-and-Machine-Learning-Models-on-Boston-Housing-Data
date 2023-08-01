# Machine Learning Algorithms for Boston Housing Data

Comprehensive analysis of Boston Housing Data aimed at predicting median housing prices using Machine Learning methods such as Regression Models, Variable Selection, Regression Trees, Bagging, Random Forest, Boosted Regression Trees, Generalized Additive Models, and Neural Networks. The in-sample and out-of-sample prediction accuracy and performance of these models are explored.

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


Note that a total of 1,000 trees is considered for Bagging, Random Forest, and Boosting. 

Table 1 reveals that the trees perform better both in-sample and out-of-sample compared to the linear regression model. Meanwhile, the best regression tree which is a pruned tree of size 10, performs worse than the other tree models. This is because the other procedures are based on the averages of multiple trees on bootstrap samples (Bagging and Bootstrap) and sequential learning (Boosting). 

Meanwhile, the advanced tree methods: Bagging, Boosting, and Random Forest perform better than the GAM model for both in-sample and out-of-sample. However, the GAM model considers the non-linear relationship of the predictor variables with the response variable. 

Overall, the neural network has the best prediction performance. This method has the best out-of-sample performance compared to all the models considered. In conclusion, the neural network does a much better job at predicting compared to linear regression, tree models, and generalized additive model.
