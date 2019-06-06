from utils.db_client import DBClient
from analysis.sentiment_analysis import Pipeline
from datetime import datetime as dt
from ast import literal_eval
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# import NB and LR classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os
from datetime import datetime as dt
import pathlib, os
import joblib
import time
import json
import argparse
import logging

def run(config_file):
    with open(config_file) as json_file:  
        params = json.load(json_file)

    iteration_name = params['iteration_name']

    # Logging information about the run
    time_now = dt.now()
    log_file_name = "log_exec_model_build__{}_{}.txt".format(iteration_name, time_now)
    logging.basicConfig(filename = log_file_name, filemode = "w", level = logging.DEBUG,
                        format = "%(asctime)s - %(levelname)s - %(message)s")

    log_init_message = ("Logging a model-build run on {}, " +
                        "with experiment name {} " +
                        "and model run type {}").format(
                            time_now, iteration_name, params['run_type']
                        )
    logging.info(log_init_message)

    # PART 1: DATA LOAD

    # DB Query Params

    # name of the feature table
    feature_table_name = params['feature_table_name']
   
    # list of feature columns
    reqd_cols = params['reqd_cols']

    label_col = params['label_col']# enter name of the label column

    # Logging data load parameters
    log_dl_msg = ("The feature table used for this run is {}\n" +
                  "The required set of columns used for this run is: \n{}\n" +
                  "The column used as the label (predicted) column for this run is {}\n").format(
                      feature_table_name, reqd_cols, label_col)
    logging.info(log_dl_msg)

    # Data Prep Params
    test_frac = float(params['test_frac']) # Fraction of the dataset to be used as a testing set
    val_frac = float(params['val_frac']) # Fraction of the dataset to be used as a testing set
    
    train_frac = 1 - test_frac
    val_frac = val_frac/train_frac #calculate updated val_frac for use in two-step split below
    reqd_cols = params['reqd_cols']
    label_col = params['label_col']

    # Logging data prep parameters
    log_dp_msg = ("The percentage of data used as testing fraction is {}\n" +
        "The percentage of data used as validation fraction is {}").format(test_frac, val_frac)
    logging.info(log_dp_msg)

    # Data Prep Objects
    db_client = DBClient()
    selected_col_query_section = ", ".join(reqd_cols)

    try:
        # TO_DO
        # change back after testing
        load_query = "SELECT {} FROM {} ORDER BY RANDOM() LIMIT 500000;".format(selected_col_query_section, 
                                                        feature_table_name)

        rows = db_client.read(statement = load_query)
        data = pd.DataFrame(rows)
        data.columns = reqd_cols
        log_msg = "STEP PASSED: LOADED DB DATA INTO MEMORY" 
        print(log_msg)
        logging.debug(log_msg)
    except Exception as e:
        log_msg = "DATALOAD ERROR: UNABLE TO LOAD THE DATA FROM THE FEATURES TABLE" 
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)
        
    try:
        data['tweet_text_clean'] = data['tweet_text_clean'].apply(lambda x: literal_eval(x))
        data['len'] = data['tweet_text_clean'].apply(lambda x: len(x))
        data = data[data['len'] > 0]
        data.loc[data.label == 4, 'label'] = 1
        data['label'] = data['label'].astype(int)
        print(data.label.unique())
        data.to_csv('analysis/models_store/sample_{}_{}.csv'.format(time_now, iteration_name))

        X, y = (data.loc[:, data.columns != label_col], data[label_col])
        log_msg = "STEP PASSED: SUCCESSFULLY SPLIT X AND Y COLS" 
        print(log_msg)
        logging.debug(log_msg)
    except Exception as e:
        log_msg = "DATAPREP ERROR: UNABLE TO SPLIT DATA INTO X AND Y SETS"
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)

    try:
        X['doc'] = X['tweet_text_clean'].apply(lambda x: " ".join(x))
        print(X.head())
        min_df = .0001
        logging.debug("The min_df used is {}".format(min_df))
        vectorizer = TfidfVectorizer(min_df = min_df)
        X = vectorizer.fit_transform(X['doc'])
        print(X.shape)
        print(y.shape)
    except Exception as e:
        log_msg = "DATAPREP ERROR: UNABLE TO PRODUCE TF-IDF FEATURES"
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)

    try:
        X_train_int, X_test, y_train_int, y_test = train_test_split(X, y, test_size = test_frac, random_state = 1234, stratify = y)
        print("int shapes...")
        print(X_train_int.shape)
        print(y_train_int.shape)
        print("splitting into val set...")
        X_train, X_val, y_train, y_val = train_test_split(X_train_int, y_train_int, test_size = val_frac, random_state = 1234, stratify = y_train_int)
        X_train = X_train.todense()
        X_val = X_val.todense()
        X_test = X_test.todense()

        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)    
        log_msg = "STEP PASSED: SUCCESSFULLY SPLIT DATA INTO TEST, VALIDATION, AND TRAIN SETS"
        print(log_msg)
        logging.debug(log_msg)
    except Exception as e:
        log_msg = "DATAPREP ERROR: UNABLE TO SPLIT DATA INTO TESTING, VALIDATION, AND TRAINING SET"
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)
    
    # PART 2: MODEL BUILD

    # Model pipeline params
    run_type = params['run_type']
    model_obj_path = params['model_obj_path']
    scoring = params['scoring']
    grid = params['grid']

    # Logging the modelling parameters
    log_mdl_msg = ("The path of the model object used for this run is found at \n{}\n" +
                   "The scoring function used for this run is: {}\n" +
                   "The parameter grid used for the search is: \n{}\n").format(
                       model_obj_path, scoring, grid)
    logging.info(log_mdl_msg)

    for key in grid.keys():
        print("Now training model family {} ".format(key))
        try:
            pipeline = Pipeline(pipeline_mode = run_type, grid_model_id_key= key, X_train = X_train, 
            y_train = y_train, clf_grid = grid, model_obj_path = model_obj_path, model_obj_pref=iteration_name, scoring = "accuracy")
            y_val_pred_class = pipeline.gen_preds(X_val)
            recall =recall_score(y_val, y_val_pred_class)
            precision = precision_score(y_val, y_val_pred_class) 
            accuracy = accuracy_score(y_val, y_val_pred_class)

            print("Validation precision: {}".format(precision))
            print("Validation recall: {}".format(recall))
            print("Validation accuracy: {}".format(accuracy))
            logging.info("Validation precision: {}".format(precision))
            logging.info("Validation recall: {}".format(recall))
            logging.info("Validation accuracy: {}".format(accuracy))

            y_test_pred_class = pipeline.gen_preds(X_test)
            recall_test =recall_score(y_test, y_test_pred_class)
            precision_test = precision_score(y_test, y_test_pred_class) 
            accuracy_test = accuracy_score(y_test, y_test_pred_class)

            print("Test precision: {}".format(precision_test))
            print("Test recall: {}".format(recall_test))
            print("Test accuracy: {}".format(accuracy_test))
            logging.info("Test precision: {}".format(precision_test))
            logging.info("Test recall: {}".format(recall_test))
            logging.info("Test accuracy: {}".format(accuracy_test))
            

        except Exception as e:
            log_msg = "MODEL BUILD ERROR: MODEL FAILED"
            print(e)
            logging.error(log_msg)
            logging.error(e)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()

    run(args.config)

