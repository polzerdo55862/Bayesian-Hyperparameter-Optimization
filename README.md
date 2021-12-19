# Bayesian-Hyperparameter-Optimization

**Table of content**




The performance of a machine learning method depends on the used data set and the chosen hyperparameter settings for the model training. Finding the optimal hyperparameter settings is crucial for building the best possible model for the data at hand.
The figure below shows a simple 2-dimensional dataset for a regression problem. In this example Polynomial Regression is used to build a predictive model. The model complexity can be determined in advance via the polynomial degree.

……..

To evaluate the performance of the model for various hyperparameter settings a suitable loss function needs to be defined. An often used cost function for regression problems is the Mean Squared Error (MSE):


<img src="/img/loss_function.png" style="width:100%">
<figcaption align = "center"><b>Loss Function: Mean Squared Error (MSE) - Image by the author</b></figcaption>

The performance of the machine learning estimator depends on the hyperparameters and the dataset used for training and validation. In order to obtain a generalised assessment of the performance of the algorithm used, the statistical procedure k-fold cross-validation (CV) is used in the following. 

Therefor the data set is split in k subsets. Following k-1 subsets are used as training data set, one for validation. After the model was build, the MSE for the validation data set is calculated. This procedure is repeated until each subset has been used once as a validation data set. The CV score is then calculated as the average of the MSE values.

<img src="/img/cross_val_score.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

So the target of hyperparameter optimization in this case would be to find the optimal hyperparameter settings, 
where the Loss (e.g. the CV Score) is minimal. Since the analytical form of the function CV is not given, 
we speaking about a so called Black Box Function. 



