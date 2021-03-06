import pandas as pd
import numpy as np
import re
import sys
import time
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import RobustScaler, StandardScaler


def define_clfs_params(grid_size):

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
            'DT': DecisionTreeClassifier()
            }

    large_grid = {
        'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']}
           }

    small_grid = {
        'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']}
           }

    test_grid = {
        'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
        'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
        'SVM' :{'C' :[0.01],'kernel':['linear']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def recall_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    return recall_score(y_true, preds_at_k, average='binary')

def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def accuracy_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    return accuracy_score(y_true, preds_at_k)

def f1_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    return f1_score(y_true, preds_at_k, average='binary')

def get_feature_importance(clf, model_name):
    clfs = {'RF':'feature_importances',
            'SVM': 'coef',
            'DT': 'feature_importances'
            }

    if clfs[model_name] == 'feature_importances':
        return  list(clf.feature_importances_)
    elif clfs[model_name] == 'coef':
        return  list(clf.coef_.tolist())
    else:
        return None


def clf_loop(models_to_run, clfs, grid, X_train, y_train, X_test, y_test):
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'time_used',
                                        'auc-roc','accuracy',
                                        'p_at_5','p_at_10', 'p_at_30', 'p_at_50',
                                        'r_at_5','r_at_10','r_at_30', 'r_at_50',
                                        'a_at_5','a_at_10','a_at_30', 'a_at_50',
                                        'f1_at_5','f1_at_10','f1_at_30', 'f1_at_50',
                                        'feature_importance', 'col_used_for_feat_importance'))

    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index])
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)
                start_time = time.time()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_probs = list(clf.predict_proba(X_test)[:,1])
                end_time = time.time()
                elapsed_time = end_time - start_time
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                results_df.loc[len(results_df)] = [models_to_run[index],clf, p, elapsed_time,
                                                   roc_auc_score(y_test, y_pred_probs),
                                                   accuracy_score(y_test,y_pred),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   accuracy_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   accuracy_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                   f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                   get_feature_importance(clf, models_to_run[index]),
                                                   list(X_train.columns.values)]

            except IndexError:
                print("IndexError")
                continue
    return results_df


def go(X_train, y_train, X_test, y_test, models=['DT','RF','SVM'], grid_size='test', result_file='report.pkl'):
    clfs, grid = define_clfs_params(grid_size)
    results_df = clf_loop(models, clfs, grid, X_train, y_train, X_test, y_test)
    results_df.to_pickle(result_file)
    return results_df
