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

# hyperparamter settings
epsilon_samples = [2, 2, 2]
C_samples = [1, 5, 10]

#helper_functions.plot_sample_svm_models(epsilon_samples, C_samples, X, y)

#######################################################################################################################
# Define hyperparameter space and plot 3d-hyperparameter-evaluation plot
#######################################################################################################################

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

#helper_functions.hyperparameter_evaluation_3d_plot(epsilon_min, epsilon_max, C_min, C_max, step, X, y)

#######################################################################################################################
# 2d plot - Neg. Cross Val. Score
#######################################################################################################################

# calculate cv_score for each hyperparameter combination
epsilon_min = 3.5
epsilon_max = 6
C_min = 7
C_max = 7.0001
step = 0.01

#helper_functions.plot_2d_evaluation_plot(epsilon_min, epsilon_max, C_min, C_max, step, X, y)

#######################################################################################################################
# Baysian Optimization
#######################################################################################################################

def baysian_optimization():
    # read black-box function values
    df = pd.read_csv (r'.\data\hyperparameter_evaluation_2d_epsilon=3.5-6_C=7-7.0001.csv')

    #pick random sample points from the black-box function for training the GP
    select_sample_indexes = [19, 40, 60, 90, 200, 220]
    y_train_sample = []
    X_train_sample = []

    for i in select_sample_indexes:
        y_train_sample.append(df.cv_scores.tolist()[i])
        X_train_sample.append(df.epsilon_setting.tolist()[i])

    X_train_sample = np.array(X_train_sample)
    X_train_sample = X_train_sample.reshape(-1, 1)

    # use calculated values of the black-box function for training
    y_black_box = df.cv_scores.tolist()

    #X_black_box = df.epsilon_setting.tolist()
    X_black_box = np.array(df.epsilon_setting)
    X_black_box = X_black_box.reshape(-1, 1)

    # create plots of prior and posteriori Gaussian Process
    helper_functions.plot_gaussian_process(X_train_sample,
                                           y_train_sample,
                                           X_black_box,
                                           y_black_box,
                                           x_min=3.5,
                                           x_max=6.0,
                                           y_min=min(y_train_sample)-1,
                                           y_max=max(y_train_sample)+1)

    #helper_functions.grid_search(epsilon_list, C_list, X, y)

    #helper_functions.calculate_expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01)

baysian_optimization()