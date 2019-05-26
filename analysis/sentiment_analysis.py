import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_sample_weight 
from sklearn.model_selection import GridSearchCV
import graphviz
import pickle
from datetime import datetime as dt
import logging
import sys
import matplotlib
matplotlib.use('PS')
from matplotlib import pyplot as plt
import pathlib, os
import joblib
 


logger = logging.getLogger('model_log')
sh = logging.StreamHandler(sys.stdout)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

class Pipeline():

    def __init__(self, pipeline_mode, grid_model_id_key = None, 
                 X_train = None, y_train = None,
                 clf_grid = None, model_obj_pref = "", scoring = "recall", 
                 threshold = 2000, threshold_type = "count", model_obj_path = ""):
        
        """
        The pipeline class is used for the following three functions:
            1. Find the optimal classifier to run given a hyperparameter 
               grid (build mode)
            2. Refresh a pre-determined classifier with new data (refresh mode)
            3. Load a pre-trained classifier and predict on new data (load mode)

        The pipeline class contains methods to calculate accuracy, recall, 
        precision, and f1 evaluation metrics at a user-specified threshold.

        Running the pipeline in build mode also a generates a joblib dump of 
        the model.
        
        :param pipeline_mode: The type of pipeline object to be initialised. Can
                              take options as "build", "refresh" and "load", 
                              Class description explains these different modes.
        :type pipeline_mode: string
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
        :param threshold: Threshold number of prediction at which scoring metric
                          needs to be calculated
        :type threshold: int, optional
        """
        self.X_train = X_train
        self.y_train = y_train
        self.clf_grid = clf_grid 
        self.model_obj_pref = model_obj_pref 
        self.scoring = scoring
        if threshold_type == "count":
            self.threshold = threshold
        else:
            self.threshold = int(len(y_train) * (threshold / 100.0))
        if pipeline_mode.lower() == "build":
            scorer = self._make_score_fxn()
            grid_obj = self._train_grid(scorer = scorer, key = grid_model_id_key)
            self._estimator = grid_obj.best_estimator_
        elif pipeline_mode.lower() == "load":
            self._estimator = self._load_model_obj(model_obj_path = model_obj_path)

    @property
    def estimator(self):
        return self._estimator

    def _make_score_fxn(self):
        """
        Private wrapper function to either create a precision or recall scorer
        function to be used in the model training process

        :return: sklearn scorer object that returns a scalar score; greater is better
        :rtype: scorer object
        """

        if self.scoring == "precision":

            def precision_at_k(y_test, y_pred_probs):
                """Calculate precision of predictions at a threshold k.
                
                :param y_test: labels for testing data
                :type y_test: array
                :param y_pred_probs: predicted probabilities of class = 1 for testing data
                :type y_pred_probs: array
                :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
                :type k: int
                :return: precision of model at k%
                :rtype: float
                """
                
                idx = np.argsort(np.array(y_pred_probs), kind='mergesort')[::-1]
                y_pred_probs_sorted = np.array(y_pred_probs)[idx]
                y_test_sorted = np.array(y_test)[idx]
                preds_at_k = [1 if x < self.threshold else 0 for x in range(len(y_pred_probs_sorted))]
                precision = precision_score(y_test_sorted, preds_at_k)
                
                return precision

            return make_scorer(precision_at_k)

        else:

            def recall_at_k(y_test, y_pred_probs):
                """Calculate recall of predictions at a threshold k.
                
                :param y_test: labels for testing data
                :type y_test: array
                :param y_pred_probs: predicted probabilities of class = 1 for testing data
                :type y_pred_probs: array
                :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
                :type k: int
                :return: recall of model at k%
                :rtype: float
                """

                idx = np.argsort(np.array(y_pred_probs), kind='mergesort')[::-1]
                y_pred_probs_sorted = np.array(y_pred_probs)[idx]
                y_test_sorted = np.array(y_test)[idx]
                preds_at_k = [1 if x < self.threshold else 0 for x in range(len(y_pred_probs_sorted))]
                recall = recall_score(y_test_sorted, preds_at_k)

                return recall

            return make_scorer(recall_at_k)

    def _train_grid(self, scorer, key):       
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
        clf = GridSearchCV(model, parameters, scoring = scorer, cv=5)
        clf.fit(self.X_train, self.y_train)
        time_now = dt.now()
        filepath_base = os.path.join(pathlib.Path(__file__), "analysis/models_store")

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

    def _model_refresh(self, model_obj_path, X_train, y_train):
        """
        Private function to execute the "model refresh" pipeline. Activities
        in this pipeline are:
            1. Load a pre-built pipeline
            2. Train the pipeline
        
        :param model_obj_path: Path of the pickled sklearn pipeline object
                               generated in a previous run
        :type model_obj_path: string
        :param X_train: Training regressor dataset to train classifier, must 
                        follow format required by scikit models
        :type X_train: pandas df/numpy array
        :param y_train: Training dependent dataset to train classifier, must 
                        follow format required by scikit models
        :type y_train: pandas series/numpy array
        :return: The retrained sklearn pipeline object
        :rtype: sklearn.pipeline
        """

        classifier_obj = self._load_model_obj(model_obj_path)
        classifier_obj = classifier_obj.fit(X_train, y_train)

        time_now = dt.now()      
        filepath_base = os.path.join(pathlib.Path(__file__).parent, "models_store")  
        model_obj_file_name = '{}_{}.joblib'.format(self.model_obj_pref, time_now)
        filepath = os.path.join(filepath_base, model_obj_file_name)
        joblib.dump(classifier_obj, filepath)

        return classifier_obj 

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
        # [:, 1] returns probability class = 4
        return(self.estimator.predict_proba(Xtest)[:, 1])

    def _get_sample_weights(self, if_balance_weights, y_train):
        """
        Private function to calculate sample weights to balance classes
        in the training dataset.
        
        :param if_balance_weights: Flag which if true, calculates sample 
                                   weights to balance out imbalanced classes
        :type if_balance_weights: bool
        :param y_train: Training dependent dataset to train classifier, must 
                        follow format required by scikit models
        :type y_train: pandas series/numpy array
        :return: The sample weights to be used by the sklearn fit function. 
                 Can be an array of weights or None
        :rtype: numpy array
        """

        sample_weights = None
        if if_balance_weights:
            sample_weights = compute_sample_weight("balanced", y_train)
    
        return sample_weights

    def joint_sort_descending(self, l1, l2):
        """
        Jointly sort two numpy arrays, util for methods below
        
        :param l1: list one
        :type l1: numpy array
        :param l2: list two
        :type l2: numpy array
        :return: sorted lists
        :rtype: numpy arrays
        """

        idx = np.argsort(l1, kind='mergesort')[::-1]
        return l1[idx], l2[idx]

    ##### Model Evaluation Functions #####
    def generate_binary_at_k(self, y_pred_probs, k, k_type):
        """
        Turn probabilistic outcomes to binary variable based on threshold k. 
        k can be either a number threshold or a percentage of total threshold
        
        :param y_pred_probs: predicted probabilities of class = 1 for testing data
        :type y_pred_probs: array
        :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
        :type k: int
        :param k_type:Type of threshold K is - either absolute number or percentage. 
                      Can take values "percent" or "count"
        :type k_type: str
        :return: array of binary classifications
        :rtype: array
        """


        if k_type.lower() == "percent":
            cutoff_index = int(len(y_pred_probs) * (k / 100.0))
        elif k_type.lower() == "count":
            cutoff_index = k
        
        # positive sentiment = 4
        test_predictions_binary = [
            4 if x < cutoff_index else 0 for x in range(len(y_pred_probs))]

        return test_predictions_binary

    def precision_at_k(self, y_test, y_pred_probs, k, k_type):
        """
        Calculate precision of predictions at a threshold k.
        k can be either a number threshold or a percentage of total threshold
        
        :param y_test: labels for testing data
        :type y_test: array
        :param y_pred_probs: predicted probabilities of class = 1 for testing data
        :type y_pred_probs: array
        :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
        :type k: int
        :param k_type:Type of threshold K is - either absolute number or percentage. 
                      Can take values "percent" or "count"
        :type k_type: str
        :return: precision of model at k%
        :rtype: float
        """

        y_pred_probs, y_test = self.joint_sort_descending(
            np.array(y_pred_probs), np.array(y_test))
        preds_at_k = self.generate_binary_at_k(y_pred_probs, k, k_type)
        precision = precision_score(y_test, preds_at_k)

        return precision

    def recall_at_k(self, y_test, y_pred_probs, k, k_type):
        """
        Calculate recall of predictions at a threshold k.
        k can be either a number threshold or a percentage of total threshold.

        :param y_test: labels for testing data
        :type y_test: array
        :param y_pred_probs: predicted probabilities of class = 1 for testing data
        :type y_pred_probs: array
        :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
        :type k: int
        :param k_type:Type of threshold K is - either absolute number or percentage. 
                      Can take values "percent" or "count"
        :type k_type: str
        :return: recall of model at k%
        :rtype: float
        """

        y_pred_probs_sorted, y_test_sorted = self.joint_sort_descending(
            np.array(y_pred_probs), np.array(y_test))
        preds_at_k = self.generate_binary_at_k(y_pred_probs_sorted, k, k_type)
        recall = recall_score(y_test_sorted, preds_at_k)

        return recall

    def f1_at_k(self, y_test, y_pred_probs, k, k_type):
        """
        Calculate F1 score of predictions at a threshold k.
        k can be either a number threshold or a percentage of total threshold

        :param y_test: labels for testing data
        :type y_test: array
        :param y_pred_probs: predicted probabilities of class = 1 for testing data
        :type y_pred_probs: array
        :param k: percentage cutoff to calcualte binary (eg. top 20% proabilities = 1)
        :type k: int
        :param k_type:Type of threshold K is - either absolute number or percentage. 
                      Can take values "percent" or "count"
        :type k_type: str
        :return: f1 score of modek at k%
        :rtype: float
        """

        y_pred_probs, y_test = self.joint_sort_descending(
            np.array(y_pred_probs), np.array(y_test))
        preds_at_k = self.generate_binary_at_k(y_pred_probs, k, k_type)

        f1 = f1_score(y_test, preds_at_k)

        return f1

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