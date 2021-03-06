#import required libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import customized helper functions
import helper_functions
import os

#load the dataset 'Boston housing'
X, y = helper_functions.read_boston_housing()

#######################################################################################################################
# Plot some sample SVM regression models
#######################################################################################################################

def create_sample_svm_regression_plots():
    # hyperparamter settings
    epsilon_samples = [2, 2, 2]
    C_samples = [1, 5, 10]

    helper_functions.plot_sample_svm_models(epsilon_samples, C_samples, X, y)

#######################################################################################################################
# Plot sample SVM Regression Model for boston housing data set
#######################################################################################################################

def plot_sample_svm_regression_model():
    # test data set
    epsilon = [2, 2, 2]
    C = [1, 5, 10]

    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    # plt.setp(ax)

    for i in range(len(C)):
        X_test, y_test, scores = helper_functions.cross_validation_svm_regression(X, y, C[i], epsilon[i])
        plt.plot(X_test, y_test, label="Model (C = " + str(C[i]) + ", Epsilon = " + str(
            epsilon[i]) + ") - Cross val. score / MSE: " + str(round(-scores.mean(), 2)))

    plt.scatter(X, y, edgecolor='black', color='grey', s=20, label="Data set")
    plt.xlabel("LSTAT [%]")
    plt.ylabel("MEDV [$1000]")

    plt.legend(loc="best")
    # plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
    #    degrees[i], -scores.mean(), scores.std()))
    plt.show()

    plt.savefig("polynomial_regression_example.png", dpi=150)

#######################################################################################################################
# Define hyperparameter space and plot 3d-hyperparameter-evaluation plot
#######################################################################################################################

def create_3d_evaluation_plot():
    # define the hyperparameter space
    hyperparameter_space = 4

    if hyperparameter_space == 1:
        epsilon_min = 1
        epsilon_max = 20
        C_min = 1
        C_max = 20
        step = 1

    if hyperparameter_space == 2:
        epsilon_min = 1
        epsilon_max = 8
        C_min = 1
        C_max = 20
        step = 1

    if hyperparameter_space == 3:
        epsilon_min = 1
        epsilon_max = 12
        C_min = 20
        C_max = 20
        step = 0.1

    if hyperparameter_space == 4:
        epsilon_min = 0.1
        epsilon_max = 30
        C_min = 0.1
        C_max = 50
        step = 1

    helper_functions.hyperparameter_evaluation_3d_plot(epsilon_min, epsilon_max, C_min, C_max, step, X, y)

#######################################################################################################################
# 2d plot - Neg. Cross Val. Score
#######################################################################################################################

def create_2d_evaluation_plot():
    # calculate cv_score for each hyperparameter combination
    helper_functions.plot_2d_evaluation_plot(epsilon_min = 0.01,
                                             epsilon_max = 15,
                                             C_min = 7,
                                             C_max = 7.0001,
                                             step = 0.01,
                                             X = X,
                                             y = y)

#######################################################################################################################
# Baysian Optimization
#######################################################################################################################

# epsilon_list=list(np.arange(epsilon_min, epsilon_max, step))
# C_list = list(np.arange(C_min, C_max, step))
#
# # calculate cv_score for each hyperparameter combination
# cv_scores, c_settings, epsilon_settings = helper_functions.grid_search(epsilon_list, C_list, X, y)
# passv


def bayesian_optimization(sample_iterations, C_fix, Epsilon_initial_sample):

    # read black-box function values
    parent_path = os.path.dirname(os.path.abspath(__file__))
    file = parent_path + r'\data\hyperparameter_evaluation_2d_epsilon=0.01-15_C=7-7.0001.csv'
    df = pd.read_csv (file)

    # define a random first sample point
    C_list =[C_fix]
    Epsilon_list = [Epsilon_initial_sample]

    X_train_sample, y_train_sample = helper_functions.generate_train_data_set(C_list, Epsilon_list, X, y)

    # use calculated values of the black-box function to compare with GP
    y_black_box = df.cv_scores.tolist()

    #X_black_box = df.epsilon_setting.tolist()
    X_black_box = np.array(df.epsilon_setting)
    X_black_box = X_black_box.reshape(-1, 1)

    for i in range(0,sample_iterations,1):
        # create plots of prior and posteriori Gaussian Process
        x_next_sample_point = helper_functions.plot_gaussian_process(X_train_sample,
                                               y_train_sample,
                                               X_black_box,
                                               y_black_box,
                                               x_min=0.1,
                                               x_max=15,
                                               y_min=min(y_train_sample)-50,
                                               y_max=max(y_train_sample)+50)

        # append new sampling point for Espilon to train data set and train GP model again
        Epsilon_list.append(x_next_sample_point)
        X_train_sample, y_train_sample = helper_functions.generate_train_data_set(C_list, Epsilon_list, X, y)


#######################################################################################################################
# Execute defined functions to create plots
#######################################################################################################################

#plot_sample_svm_regression_model()
#create_sample_svm_regression_plots()
#create_3d_evaluation_plot()
#create_2d_evaluation_plot()

bayesian_optimization(sample_iterations=8, C_fix = 7, Epsilon_initial_sample = 3.5)