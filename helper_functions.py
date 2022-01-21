#import required libaries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import read_csv
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import numpy as np
import os

######################################################################################################
# Boston Housing Dataset
######################################################################################################
#  The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
#  prices and the demand for clean air', J. Environ. Economics & Management,
#  vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
#  ...', Wiley, 1980.   N.B. Various transformations are used in the table on
#  pages 244-261 of the latter.
#
#  Variables in order:
#  CRIM     per capita crime rate by town
#  ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  INDUS    proportion of non-retail business acres per town
#  CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  NOX      nitric oxides concentration (parts per 10 million)
#  RM       average number of rooms per dwelling
#  AGE      proportion of owner-occupied units built prior to 1940
#  DIS      weighted distances to five Boston employment centres
#  RAD      index of accessibility to radial highways
#  TAX      full-value property-tax rate per $10,000
#  PTRATIO  pupil-teacher ratio by town
#  B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  LSTAT    % lower status of the population
#  MEDV     Median value of owner-occupied homes in $1000's
# ######################################################################################################

def read_boston_housing():
    '''
    Reads boston housing data from local csv file
    Returns:
    -------
    X,y: defined data set
    '''
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    parent_path = os.path.dirname(os.path.abspath(__file__))
    file = parent_path + r'\data\boston_housing.csv'
    data = read_csv(file, header=None, delimiter=r"\s+", names=column_names)
    print(data.head(5))

    # LSTAT: % lower status of the population
    X = data['LSTAT']
    X = np.array(X)

    # MEDV: Median value of owner-occupied homes in $1000's
    y = data['MEDV']

    return X,y

def grid_search(epsilon_list, C_list, X, y):
    '''

    Parameters
    ----------
    epsilon: Hyperparameter 1 SVM Regression
    C: Hyperparameter 2 SVM Regression
    X: Dataset
    y: Dataset

    Returns
    -------
    cv_scores: list with the cv score for each evaluated hyperparameter combination
    c_settings: used settings for Hyperparameter "C"
    epsilon_settings: used settings for Hyperparameter "epsilon"

    '''

    cv_scores = []
    c_settings = []
    epsilon_settings = []

    for n in range(len(epsilon_list)):
        for i in range(len(C_list)):

            # support vector regression
            pipeline = make_pipeline(StandardScaler(), SVR(C=C_list[i], epsilon=epsilon_list[n]))
            pipeline.fit(X[:, np.newaxis], y)

            # Evaluate the models using cross-validation
            scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
            cv_scores.append(round(scores.mean(), 2))
            c_settings.append(C_list[i])
            epsilon_settings.append(epsilon_list[n])

    return cv_scores, c_settings, epsilon_settings

def cross_validation_svm_regression(X, y, C, epsilon):
    '''

    Parameters
    ----------
    X: train data set
    epsilon: Hyperparameter 1 SVM Regression
    C: Hyperparameter 2 SVM Regression

    Returns
    -------
    X_test: list with X_test data (here: X_test = LSTAT)
    y_test: list with predicted values (here: y=MEDV)
    '''


    # support vector regression
    pipeline = make_pipeline(StandardScaler(), SVR(C=C, epsilon=epsilon))
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)
    print("Cross Validation Score for models with a polynomial degree of " + str(epsilon) +
          ": MSE = " + str(round(-scores.mean(), 2)))

    X_test = np.linspace(0, max(X), 1000)
    y_test = pipeline.predict(X_test[:, np.newaxis])

    return X_test, y_test, scores

def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 0.5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-0.1, 0.1])

def plot_sample_svm_models(epsilon_samples, C_samples, X, y):

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    # plt.setp(ax)

    for i in range(len(C_samples)):
        X_test, y_test, scores = cross_validation_svm_regression(X, y, C_samples[i], epsilon_samples[i])
        plt.plot(X_test, y_test,
                 label="Model (C = " + str(C_samples[i]) + ", Epsilon = " + str(epsilon_samples[i]) + ") - Cross val. score / MSE: " + str(
                     round(-scores.mean(), 2)))

    plt.scatter(X, y, edgecolor='black', color='grey', s=20, label="Data set")
    plt.xlabel("LSTAT [%]")
    plt.ylabel("MEDV [$1000]")

    plt.legend(loc="best")
    # plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
    #    degrees[i], -scores.mean(), scores.std()))
    plt.savefig(r"./plots/polynomial_regression_example.png", dpi=150)
    plt.show()

def hyperparameter_evaluation_3d_plot(epsilon_min, epsilon_max, C_min, C_max, step, X, y):
    '''

    Parameters
    ----------
    Hyperparameter space:
        epsilon_min
        epsilon_max
        C_min
        C_max
    step: step size
    X
    y

    Returns:
    save plot to ./plots
    -------

    '''
    ngridx = 100
    ngridy = 200
    npts = 200

    epsilon_list = list(np.arange(epsilon_min, epsilon_max, step))
    C_list = list(np.arange(C_min, C_max, step))

    # calculate cv_score for each hyperparameter combination
    cv_scores, c_settings, epsilon_settings = grid_search(epsilon_list, C_list, X, y)

    # define plot dimensions
    x_plot = c_settings
    y_plot = epsilon_settings
    z_plot = cv_scores

    # define figure
    fig, ax1 = plt.subplots(1)

    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.
    xi = np.linspace(min(x_plot) - 1, max(x_plot) + 1, ngridx)
    yi = np.linspace(min(y_plot) - 1, max(y_plot) + 1, ngridy)

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x_plot, y_plot)
    interpolator = tri.LinearTriInterpolator(triang, z_plot)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Note that scipy.interpolate provides means to interpolate data on a grid
    # as well. The following would be an alternative to the four lines above:
    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr1, ax=ax1)
    ax1.plot(x_plot, y_plot, 'ko', ms=3)
    ax1.set(xlim=(min(x_plot), max(x_plot)), ylim=(min(y_plot), max(y_plot)))
    # ax1.set_title('grid and contour (%d points, %d grid points)' %
    #              (npts, ngridx * ngridy))
    ax1.set_xlabel('C')
    ax1.set_ylabel('Epsilon')

    # plt.subplots_adjust(hspace=0.5)
    plt.savefig(r"./plots/hyperparameter_evaluation_3d.png", dpi=150)
    plt.show()

def plot_2d_evaluation_plot(epsilon_min, epsilon_max, C_min, C_max, step, X, y):
    '''

    Parameters
    ----------
    epsilon_min: lower limit hyperparameter space
    epsilon_max: upper limit hyperparameter space
    C_min: lower limit hyperparameter space
    C_max: upper limit hyperparameter space
    step
    X
    y

    Returns
    -------

    '''
    epsilon_list = list(np.arange(epsilon_min, epsilon_max, step))
    C_list = list(np.arange(C_min, C_max, step))

    # calculate cv_score for each hyperparameter combination
    cv_scores, c_settings, epsilon_settings = grid_search(epsilon_list, C_list, X, y)

    x_plot = epsilon_settings
    y_plot = cv_scores

    # define figure
    fig, ax1 = plt.subplots(1)
    #ax1.plot(x_plot, y_plot, 'ko', ms=3)
    ax1.plot(x_plot, y_plot, color='black')

    #ax1.set(xlim=(min(x_plot), max(x_plot)), ylim=(min(y_plot), max(y_plot)))
    # ax1.set_title('grid and contour (%d points, %d grid points)' %
    #              (npts, ngridx * ngridy))
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Cross-val. score')


    #calc optimal epsilon setting
    max_cv_score = max(cv_scores)
    index_max_cv_score = cv_scores.index(max_cv_score)
    epsilon_optimal = epsilon_settings[index_max_cv_score]
    ax1.axvline(x=epsilon_optimal, color = 'grey')
    ax1.axhline(y=max_cv_score, color = 'grey')

    ax1.title.set_text("C = {} / Espilon = {} - {} / Optimal Epsilon Setting: {}".format(C_min, epsilon_min, epsilon_max, epsilon_optimal))

    hyperparameter_opt_2d_df = pd.DataFrame(
        {'c_setting': c_settings,
         'epsilon_setting': epsilon_settings,
         'cv_scores': cv_scores
         })

    filename = r"./data/hyperparameter_evaluation_2d_epsilon=" + str(epsilon_min)\
        + '-' + str(epsilon_max) + '_C=' + str(C_min) + "-" + str(C_max) + ".csv"
    hyperparameter_opt_2d_df.to_csv(filename, index=False)

    # plt.subplots_adjust(hspace=0.5)
    plt.savefig(r"./plots/hyperparameter_evaluation_2d.png", dpi=150)
    plt.savefig(r"./plots/hyperparameter_evaluation_2d.svg")
    plt.show()

def model_gp_function(X_train_sample, y_train_sample):
    '''
    Learn posteriori Gaussian Process Regression model using sample data set X_train_sample

    Attributes
    -------
    X_train_sample: chosen train data set
    y_train_sample: chosen train set for target variable (here: already calculated values of the black-box function)
    Returns
    -------
    GP model
    '''

    # Choosing a kernel
    #kernel = 1.0 * RBF(length_scale=0.3, length_scale_bounds=(1e-1, 40.0))
    #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

    # define GP regressor and train model
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=2)
    gpr.fit(X_train_sample, y_train_sample)

    return gpr

def generate_train_data_set(C_list, Epsilon_list, X, y):
    '''
    Use C_list and Epsilon_list to generate a train data set for training the GP regression model, the surrogate model
    Parameters
    ----------
    x_next_sample_point
    C_list
    Epsilon_list

    Returns
    -------

    '''
    #calculate cv score for sample points
    cv_scores, c_settings, epsilon_settings = grid_search(Epsilon_list, C_list, X, y)

    #select_sample_indexes = [19, 40, 60, 90, 200, 220]
    y_train_sample = cv_scores
    X_train_sample = epsilon_settings

    # for i in select_sample_indexes:
    #     y_train_sample.append(df.cv_scores.tolist()[i])
    #     X_train_sample.append(df.epsilon_setting.tolist()[i])

    X_train_sample = np.array(X_train_sample)
    X_train_sample = X_train_sample.reshape(-1, 1)

    return X_train_sample, y_train_sample

def plot_gpr_samples(gpr_model, n_samples, ax, x_min, x_max, y_min, y_max):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(x_min, x_max, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([y_min, y_max])

    return ax


def plot_gaussian_process(X_train_sample, y_train_sample,
                          X_black_box, y_black_box,
                          x_min, x_max, y_min, y_max):

    #number of ploted sample functions for posteriori GP
    n_samples = 2

    # train the model with the already known data points from the black-box function (Cross-val. score)
    gpr = model_gp_function(X_train_sample, y_train_sample)

    # calulate expected improvement
    X, ei, X_next_sample_point = calculate_expected_improvement(X_train_sample, y_train_sample, x_min, x_max, gpr, xi=0.01)

    #fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 8))
    fig, axs = plt.subplots(nrows=2, figsize=(10, 6))

    # plot prior
    # axs[0] = plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0],
    #                           x_min=x_min, x_max=x_max,
    #                           y_min=y_min, y_max=y_max)
    #axs[0].plot(X_black_box, y_black_box)
    #axs[0].set_title("Black-box function (calculated using Grid Search)")

    # plot posterior
    gpr.fit(X_train_sample, y_train_sample)
    axs[0] = plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    axs[0].plot(X_black_box, y_black_box, label='Real Black-box function')
    axs[0].scatter(X_train_sample[:, 0], y_train_sample, color="red", zorder=10, label="Observations")
    axs[0].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    axs[0].set_xlabel("Epsilon")
    axs[0].set_ylabel("Neg. cross-val. score")
    # axs[1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper right")
    axs[0].set_title("Posteriori Gaussian Process - " + str(len(X_train_sample[:, 0])) + " sampling point(s)" )

    # fig.suptitle("Radial Basis Function kernel", fontsize=18)

    axs[1].plot(X, ei, color='black')
    axs[1].set_xlabel("Epsilon")
    axs[1].set_ylabel("Expected Improvement")
    axs[1].set_title("Expected Improvement - Next sample point: " + str(round(X_next_sample_point, 2)))
    axs[1].axvline(x=X_next_sample_point, color='red', label='Next sample')
    axs[1].legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    # axs[1].set_ylim([0,20])

    plt.tight_layout()
    plt.savefig(r"./plots/bayesian_opt_samples_" + str(len(X_train_sample)) + ".png", dpi=150)
    plt.savefig(r"./plots/bayesian_opt_samples_" + str(len(X_train_sample)) + ".svg")
    # plt.show()

    return X_next_sample_point


def calculate_expected_improvement(X_train_sample, y_train_sample, x_min, x_max, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.

    Source:
     - http://krasserm.github.io/2018/03/21/bayesian-optimization/
     - https://brendanhasz.github.io/2019/03/28/hyperparameter-optimization.html#hyperparameter-optimization
     - https://stats.stackexchange.com/questions/502590/expected-improvement-formula-for-bayesian-optimisation
     - https://arxiv.org/abs/1705.10033
    '''

    # Define n-values between x_min and x_max as X to plot EI and GP
    x = np.linspace(x_min, x_max, 100)
    X = x.reshape(-1, 1)

    # Train GP model with train data set (X_train_sample, y_train_sample)
    # - contains already calculated data points of the black-box function
    gpr = model_gp_function(X_train_sample, y_train_sample)
    mu, sigma = gpr.predict(X, return_std=True)
    #mu_sample = gpr.predict(X_train_sample)

    with np.errstate(divide='warn'):
        # mu is the mean of the distribution defined by the Gaussian process
        # sigma is the standard deviation of the distribution defined by the Gaussian process
        # norm.cdf(Z) is the standard normal cumulative density function
        # norm.pdf(Z) is the standard normal probability density function
        # xi is an exploration parameter
        Z = (mu - max(y_train_sample) - xi) / sigma
        ei = (mu - max(y_train_sample)) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # try:
        #     ei[sigma == 0.0] = 0.0
        # except:
        #     pass

    # find maximum value of expected improvement in space x_min - x_max
    ei_max = max(ei)

    # find index of max EI
    max_index = ei.tolist().index(ei_max)

    # find corresponding X value
    X_next_sample_point = X[max_index][0]

    return X, ei, X_next_sample_point