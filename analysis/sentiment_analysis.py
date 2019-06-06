import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime as dt
import logging
import sys
import matplotlib
matplotlib.use('PS')
from matplotlib import pyplot as plt
import pathlib, os
import joblib


class Pipeline():

    def __init__(self, grid_model_id_key = None, 
                 X_train = None, y_train = None,
                 clf_grid = None, model_obj_pref = "", scoring = "accuracy", 
                 model_obj_path = ""):
        
        """
        The pipeline class is used to find the optimal classifier to run given 
        a hyperparameter grid (build mode)
        
        The pipeline class contains methods to calculate accuracy, recall, 
        precision, and f1 evaluation metrics at a user-specified threshold.

        Running the pipeline in build mode also a generates a joblib dump of 
        the model.
        
        :param grid_model_id_key: The key corresponding to the model class to 
                                  be used from the input dictionary, defaults to
                                  None
        :type grid_model_id_key: string, optional
        :param X_train: Training regressor dataset to train classifier, must 
                        follow format required by scikit models. Used only in
                        build or refresh mode, defaults to None
        :type X_train: pandas df/numpy array, optional
        :param y_train: Training dependent dataset to train classifier, must 
                        follow format required by scikit models. Used only in
                        build or refresh mode, defaults to None 
        :type y_train: pandas series/numpy array, optional
        :param clf_grid: Hyperparameter grid to search over while building the new model
        :type clf_grid: Dictionary, optional
        :param model_obj_path: Path of the pickled sklearn pipeline object
                               generated in a previous run, defaults to None
        :type model_obj_path: string, optional
        :param model_obj_pref: Prefix to be appended to the name of the model
                               pickle dump, defaults to an empty string
        :type model_obj_pref: string, optional
        :param scoring: Scoring measures to be used by the optimizer. Choice of 
                        "precision" or "recall". Calculates selected metric at
                        threshold defined by parameter. Defaults to "recall"
        :type scoring: string, optional
        """
        self.X_train = X_train
        self.y_train = y_train
        self.clf_grid = clf_grid 
        self.model_obj_pref = model_obj_pref 
        self.scoring = scoring
        grid_obj = self._train_grid(key = grid_model_id_key)
        self._estimator = grid_obj.best_estimator_

    @property
    def estimator(self):
        return self._estimator

    def _train_grid(self, key):       
        """
        :param scorer: Sklearn scorer object that returns a scalar precision or recall 
                    score; greater is better
        :type scorer: scorer object
        :param key: key corresponding to classifier in grid (eg. 'DT' = 'Decision Tree')
        :type key: string
        :return: trained classifier representing best model (based on metric used by scorer)
        :rtype: sklearn classifier 
        """
        model = eval(self.clf_grid[key]["type"])
        parameters = self.clf_grid[key]["grid"]
        clf = GridSearchCV(model, parameters, scoring = self.scoring, cv=5)
        clf.fit(self.X_train, self.y_train)
        time_now = dt.now()
        filepath_base = "analysis/models_store"

        print(model)
        print(clf)
        print(clf.best_estimator_)
        print(clf.best_score_)
        
        cv_results_file_name = "{}_{}_results.csv".format(self.model_obj_pref, time_now)
        filepath = os.path.join(filepath_base, cv_results_file_name)
        df = pd.DataFrame(clf.cv_results_)
        df.to_csv(filepath)
        
        model_obj_file_name = '{}_{}.joblib'.format(self.model_obj_pref, time_now)
        filepath = os.path.join(filepath_base, model_obj_file_name)
        joblib.dump(clf.best_estimator_ , filepath)
        
        return clf

    def _load_model_obj(self, model_obj_path):
        """
        Private function to load a model joblib dump. 
        
        :param model_obj_path: filepath of dumped classifier 
        :type model_obj_path: string
        :return: Loaded pretrained sklearn pipeline object
        :rtype: sklearn.pipeline
        """

        return joblib.load(model_obj_path)

    def gen_pred_probs(self, Xtest):
        """
        Generates predicted probabilities of class membership
        
        :param Xtest: testing features
        :type Xtest: pandas dataframe
        :return: array of predicted probabilities of label = 1
        :rtype: array
        """
        # [:, 1] returns probability class = 1
        return(self.estimator.predict_proba(Xtest)[:, 1])

    def gen_preds(self, Xtest):
        """
        Generates predicted probabilities of class membership
        
        :param Xtest: testing features
        :type Xtest: pandas dataframe
        :return: array of predicted probabilities of label = 1
        :rtype: array
        """
        
        return self.estimator.predict(Xtest)

    ##### Writeout/Visualization Functions #####
    def plot_precision_recall_n(self, y_test, y_pred_prob, model_name, output_type):
        """
        Plot and output a precision recall graph for a given model run
        
        :param y_test: labels for testing data
        :type y_test: array
        :param y_pred_probs: predicted probabilities of class = 1 for testing data
        :type y_pred_probs: array
        :param model_name: model name for title of plot
        :type model_name: str
        :param output_type: if output_type = save, .png
        :type output_type: .png or none
        """

        y_score = y_pred_prob
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_score)
        precision_curve = precision_curve[:-1]
        recall_curve = recall_curve[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)
        for value in pr_thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)
        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, precision_curve, 'b')
        ax1.set_xlabel('percent of population')
        ax1.set_ylabel('precision', color='b')
        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, recall_curve, 'r')
        ax2.set_ylabel('recall', color='r')
        ax1.set_ylim([0, 1])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, 1])

        name = model_name
        plt.title(name)
        if (output_type == 'save'):
            plt.savefig(name, close=True)
        elif (output_type == 'show'):
            plt.show()
            plt.close()
        else:
            plt.show()
            plt.close()