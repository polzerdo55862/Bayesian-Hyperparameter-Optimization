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
where the Loss (e.g. the CV Score) is minimal (in the following we try find the maximum for the negative CV score). Since the analytical form of the function CV is not given, 
we speaking about a so called Black Box Function. 

<img src="/img/black_box_function.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

<img src="/img/black_box_function_evaluation.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

What we could do, is just calculate the value of f(x) for mupltiple position in the defined hyperparameter space. Like grid or random search is doing it. Afterwards we simply identify the Hyperparameter combination with the optimal value.

Definitely a valid approach, at least for so called “cheap” black-box function, where the computation effort to calculate the CV values is low. But what if the evaluation of the function pretty costly (so the computational time and/or cost to calculate CV is high)? In this case it may makes sense to think about more “intelligent” ways to find the optimal value. [Cas13]

<img src="/img/cheap_and_costly_black_box_function.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

When speaking about costly black-box functions, the outcome of interest is expensive or time-consuming to calculate, a “cheaper” surrogate function could help. 
The Surrogate Function should approximates the black-box function f(x) [Cas13].

In total, the time needed to compute the needed sample values and the surrogate function, 
should be less time-consuming than calculating each point in the hyperparameter space. 
To model the surrogate function, a wide range of machine learning techniques is used, 
like Polynomial Regression, Support Vector Machine, Neuronal Nets and probably the most 
popular, the Gaussian Process (GP).

Damit kann die Bayesian Optimization dem Feld des Active Learning zugeordnet werden. Active Learning 
tries to mimize the labeling costs. 
Ziel ist die möglichst genaue Nachbildung der Black-box function mit möglichst wenig 
Berechnungsaufwand.

Sprechen wir von Gaussian Hyperparameter Optimization, bewegen wir uns im Feld der uncertainty reduction.

As a rule, the variance is used as a measure of uncertainty. The Gaussian Process (GP) is able to map the 
the uncertainty as well. [Agn20]

For the above regression problem, the following black-box function results. 
In order to be able to map the function with sufficient accuracy for the defined 
hyperparameter space, this range must be appropriately fine-granularly ebased. 
In this case we assume a predefined hyperparameter space (polynomial degree = 1 - 100). 
Since the polynomial degree can only assume integer values, there are 30 cross validations 
to be carried out for this delimited range.

<img src="/img/evaluation_steps.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

## Surrogate Function

To reduce the number of necessary data points, 
we try to find a suitable surrogate function with few 
data points that approximates the actual course of our black-box function 
f(x) = CV(lambda). The best-known surrogate function in the context 
of hyperparameter optimisation is the Gaussian process, or more 
precisely the Gaussian process regression. A more detailed explanation 
of how the Gaussian Process Regression works can be found in "Gaussian 
Processes for Machine Learning" by Carl Edward Rasmussen and Christopher 
K. I. Williams, which is available for free at:

http://www.gaussianprocess.org/gpml/chapters/

You can also find an explanation of Gauss Process Regression in one of my recent articles:

https://towardsdatascience.com/7-of-the-most-commonly-used-regression-algorithms-and-how-to-choose-the-right-one-fc3c8890f9e3

In short, Gaussian process regression defines a priori Gaussian process that already includes prior knowledge of the 
true function. Since we usually have no knowledge about the true course of our black box function, a constant function 
with some covariance is usually freely chosen as the Priori Gauss.


....

By knowing individual data points of the true function, the possible course of the function is gradually narrowed down.


## Acquisition Function

The surrogate function is recalculated after each calculation step and serves as the basis for selecting the next calculation step. 
For this purpose, an acquisition function is introduced. The most popular acquisition function in the context of Hpyer parameter 
optimisation is the information gain.

In addition to the Expected Improvement the following Acquisition Functions are used:

- Knowledge gradient
- Entropy search 
- Predictive entropy

# References

[Agn20] Agnihotri, Apoorv; Batra, Nipun.  https://distill.pub/2020/bayesian-optimization/. 2020
[Cas13] Cassilo, Andrea. A Tutorial on Black–Box Optimization. https://www.lix.polytechnique.fr/~dambrosio/blackbox_material/Cassioli_1.pdf. 2013.
[Sci18] Sicotte, Xavier. Cross validation estimator. https://stats.stackexchange.com/questions/365224/cross-validation-and-confidence-interval-of-the-true-error/365231#365231. 2018