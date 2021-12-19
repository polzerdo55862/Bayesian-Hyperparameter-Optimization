#import required libaries
import pandas as pd
import numpy as np

# import customized helper functions
import helper_functions

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
    df = pd.read_csv (r'.\data\hyperparameter_evaluation_2d_epsilon=0.01-15_C=7-7.0001.csv')

    C_list =[C_fix]
    Epsilon_list = [Epsilon_initial_sample]

    C_list =[C_fix]
    Epsilon_list = [Epsilon_initial_sample]

    # #calculate cv score for sample points
    # cv_scores, c_settings, epsilon_settings = helper_functions.grid_search(Epsilon_list, C_list, X, y)
    #
    # #select_sample_indexes = [19, 40, 60, 90, 200, 220]
    # y_train_sample = cv_scores
    # X_train_sample = epsilon_settings

    # for i in select_sample_indexes:
    #     y_train_sample.append(df.cv_scores.tolist()[i])
    #     X_train_sample.append(df.epsilon_setting.tolist()[i])

    # X_train_sample = np.array(X_train_sample)
    # X_train_sample = X_train_sample.reshape(-1, 1)

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

        #helper_functions.grid_search(epsilon_list, C_list, X, y)

        #helper_functions.calculate_expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01)


#######################################################################################################################
# Execute defined functions to create plots
#######################################################################################################################

#create_sample_svm_regression_plots()
#create_3d_evaluation_plot()
#create_2d_evaluation_plot()
bayesian_optimization(sample_iterations=8, C_fix = 7, Epsilon_initial_sample = 3.5)

