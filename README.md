# Bayesian-Hyperparameter-Optimization

**Table of content**




The performance of a machine learning method depends on the used data set and the chosen hyperparameter settings for the model training. Finding the optimal hyperparameter settings is crucial for building the best possible model for the data at hand.
The figure below shows a simple 2-dimensional dataset for a regression problem. In this example Polynomial Regression is used to build a predictive model. The model complexity can be determined in advance via the polynomial degree.

……..

To evaluate the performance of the model for various hyperparameter settings a suitable loss function needs to be defined. An often used cost function for regression problems is the Mean Squared Error (MSE):

