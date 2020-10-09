  
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:54:25 2019
@author: yuhaili
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy import stats
from GPy.models import GPRegression
from GPy.util.normalizer import Standardize

def rmse(y_true, y_pred):
    """
    Take the true values and predicted values as input
    Return the mean squared error between y_true and y_pred
    y_true: numpy array of the true y values
    y_pred: numpy array of the predicted values
    output: mean squared error between y_true and y_pred
    """
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

def to_MPa(psi):
    """
    Take a value in psi
    Return the corresponding value in MPa
    psi: a float number in psi
    
    output: pressure in MPa
    """
    return psi*0.00689476

def read_data(file_name):
    """
    file_name: absolute path of the file to be read
    output: X, y
    """
    raw_data = pd.read_csv(file_name)
    y = raw_data['28 Day']
    X = raw_data.drop(['7 Day', '28 Day'], axis=1)
    print("Input variables:\n", X.columns)
    print("Output variables:\n", y.name)
    return X.values, y.values

def plot_result(y_true, y_pred, title):
    """
    Plot y_true vs. y_pred with y=x regression line
    y_true: numpy array of the true y values
    y_pred: numpy array of the predicted y values
    title: title name of plot
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    
    plt.xlabel("y true(MPa)")
    plt.ylabel("y pred(MPa)")
    plt.title("Result of " + title)
    
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)
    line = slope*y_true+intercept
    # now plot both limits against eachother\n
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.plot(y_true,y_pred,'o', y_true, line)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    FILE_PATH = "../results/"
    FILE_NAME = title + ".png"
    plt.savefig(FILE_PATH + FILE_NAME)
    plt.show()

def gpy_cross_validation(kern, X, y, cv=5, random_state=19, optimize=False, score=None):
    r2 = []
    rm = []
    X, y = shuffle(X, y, random_state=19)
    
    SIZE = int(len(X)/cv)
        
    for i in range(cv):
        start = i * SIZE
        end = len(X) if i == cv - 1 else (i + 1) * SIZE
        X_test, y_test, X_train, y_train = X[start: end], y[start: end], np.delete(X, slice(start, end), axis=0), np.delete(y, slice(start, end), axis=0)
        m = GPRegression(X_train, y_train, kernel=kern, normalizer=Standardize())
        if optimize:
            m.optimize()
        res, var = m.predict(X_test)
        if score is None or score == 'r2':
            r2.append(r2_score(y_test, res))
        if score is None or score == 'rmse':
            rm.append(rmse(y_test, res))
    
    return {'r2': r2, 'rmse':rm}