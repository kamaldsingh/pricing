# ig&h
import os
from datetime import datetime
from statsmodels import robust
from scipy.stats import kurtosis, skew
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import random


#sklearn 
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from xgboost import plot_importance
from xgboost import XGBClassifier
import xgboost as xgb 
from sklearn.model_selection import learning_curve, GridSearchCV, train_test_split, StratifiedKFold, validation_curve, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# SHAP and explainability
import eli5
from pdpbox.pdp import pdp_isolate, pdp_plot
from pdpbox.pdp import pdp_interact, pdp_interact_plot
import shap

# Classification 
# import plotly
# import plotly.express as px 
from xgboost import XGBClassifier, plot_importance
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, ShuffleSplit, learning_curve
from sklearn.utils import class_weight
from yellowbrick.classifier import ROCAUC
from yellowbrick.datasets import load_game

#EDA
import pandas_profiling
from pandas_profiling import ProfileReport


##############################
###### EDA and cleaning ######
##############################

def convert_boolean(x):
    """ Convert the object variables to boolean""" 
    if(x=='YES'):
        return 1
    elif(x=='NO'):
        return 0
    else:
        return np.nan
    
def encodeColums(app_train,cols_sent, cardinality_limit=1000 ):
    """ Encode all the object columns in the dataframe based on a cardinality limit""" 
    le = LabelEncoder()
    le_count = 0

    for col in cols_sent:
        if(app_train[col].dtype == 'object'):
            # If 2 or fewer unique categories
            if(len(list(app_train[col].unique())) <= cardinality_limit):
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])
                le_count += 1
            else:
                print('too many unique values to encode for:', col, np.unique(col))

    print('%d columns were label encoded.' % le_count)
    return;


def value_count_plot(df, value_col, target_col='target'):
    """Return the distribution/variation of the feature for the target columns"""
    groupeddf = df.value_counts([value_col,target_col]).reset_index()
    groupeddf.columns = [value_col,target_col,'ct']
    return(groupeddf)

####################################
###### Modelling functions #########
####################################
def plot_residuals(y_test, y_pred):
    """Plot residuals vs predicted"""
    #gather results
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    plt.scatter(df_results['Predicted'], df_results['Residuals'])
    plt.title('Residuals vs predicted values')
    plt.xlabel('Predicted Y')
    plt.ylabel('Residuals') #
    plt.show()
    return

def custom_time_split(df_model, feature_cols, target_col):
    """Time series split for given challenge"""
    df_train = df_model[df_model.contract_date<='2011-10-26']
    df_test = df_model[df_model.contract_date>'2011-10-26']
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_train = df_train[target_col]
    y_test = df_test[target_col]
    return(X_train, X_test, y_train, y_test )

def learning_curve_model_cv(X, Y, model, train_sizes):
    """Plot the Sklearn learning curve for the model with the train and test sets over different training samples """
    warnings.filterwarnings('ignore')
    plt.figure()
    plt.title("Learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, n_jobs=4, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt;