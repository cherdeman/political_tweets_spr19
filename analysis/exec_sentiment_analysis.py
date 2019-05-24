from utils.db_client import DBClient
from analysis.sentiment_analysis import Pipeline
from datetime import datetime as dt
import sklearn
# import NB and LR classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import numpy as np
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
    log_file_name = "log_exec_model_build_{}.txt".format(time_now)
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
    reqd_rows_fraction = float(params['reqd_rows_fraction']) # enter fraction of rows from feature table needed, as a decimal
    if reqd_rows_fraction < 1:
        if_nowrite_sample_tbl = False
    else:
        if_nowrite_sample_tbl = True

    # Logging data load parameters
    log_dl_msg = ("The feature table used for this run is {}\n" +
                  "The required set of columns used for this run is: \n{}\n" +
                  "The column used as the label (predicted) column for this run is {}\n" +
                  "The percentage of rows used is {}\n" +
                  "Was the sample snapshot table created in this run {}").format(
                      feature_table_name, reqd_cols, label_col, 
                      reqd_rows_fraction, not if_nowrite_sample_tbl
                  )
    logging.info(log_dl_msg)

    # Data Prep Params
    zero_downsample_frac = float(params['zero_downsample_frac']) # Fraction of the zeros you want to downsample to. Leave to 1 if not downsampling
    test_frac = float(params['test_frac']) # Fraction of the dataset to be used as a testing set
    train_frac = 1 - test_frac

    # Logging data prep parameters
    log_dp_msg = ("The Zeros in the sample were downsampled to {}%\n" +
                  "The percentage of data used as testing fraction is {}\n").format(
                      zero_downsample_frac*100, test_frac
                      )
    logging.info(log_dp_msg)

    # Data Prep Objects
    db_client = DBClient()
    data_prepper = DataPrep(db_client, feature_table_name, label_col)

    try:
        df = data_prepper.load_data(reqd_cols, if_nowrite_sample_tbl, reqd_rows_fraction, iteration_name)
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
        df = data_prepper.downsample_zeros(data = df, 
                                           frac = zero_downsample_frac)
        log_msg = "STEP PASSED: SUCCESSFULLY DOWNSAMPLED ZEROS"
        print(log_msg)
        logging.debug(log_msg)

    except Exception as e:
        log_msg = "DATAPREP ERROR: UNABLE TO DOWNSAMPLE ZEEROS. DATAFRAME PASSED \
                   WITH SAME NUMBER OF ZEROS"
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)
    
    try:
        X, y = data_prepper.split_x_y(df)
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
        X_train, X_test, y_train, y_test = data_prepper.train_test_split(X, y, test_size = test_frac,
                                                                         train_size = train_frac
                                                                         )

        print(df.shape)
        print(X.shape)
        
        log_msg = "STEP PASSED: SUCCESSFULLY SPLIT DATA INTO TEST AND TRAIN SETS"
        print(log_msg)
        logging.debug(log_msg)
    except Exception as e:
        log_msg = "DATAPREP ERROR: UNABLE TO SPLIT DATA INTO TESTING AND TRAINING SET"
        print(log_msg)
        print(e)
        logging.error(log_msg)
        logging.error(e)
    
    # PART 2: MODEL BUILD

    # Model pipeline params
    run_type = params['run_type']
    model_obj_path = params['model_obj_path']
    scoring = params['scoring']
    score_k_val = params['score_k_val']
    score_k_type = "count"
    grid = params['grid']

    # Logging the modelling parameters
    log_mdl_msg = ("The path of the model object used for this run is found at \n{}\n" +
                   "The scoring function used for this run is: {}\n" +
                   "The type of population threshold used is: {}\n" +
                   "The population threshold at which the score is calculated is {}\n" +
                   "The parameter grid used for the search is: \n{}\n").format(
                       model_obj_path, scoring, score_k_type, score_k_val, grid
                       )
    logging.info(log_mdl_msg)

    for key in grid.keys():
        try:
            pipeline = Pipeline(pipeline_mode = run_type, grid_model_id_key= key, X_train = X_train, 
            y_train = y_train, clf_grid = grid, threshold = score_k_val, threshold_type = score_k_type,
            model_obj_path = model_obj_path, model_obj_pref=iteration_name)

            y_test_prob = pipeline.gen_pred_probs(X_test)
            recall = pipeline.recall_at_k(y_test, y_test_prob, score_k_val, score_k_type)
            precision = pipeline.precision_at_k(y_test, y_test_prob, score_k_val, score_k_type) 
            y_pred_prob_ordered, y_test_ordered = pipeline.joint_sort_descending(np.array(y_test_prob), np.array(y_test))
            binary_preds = pipeline.generate_binary_at_k(y_pred_prob_ordered, score_k_val, score_k_type)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test_ordered, binary_preds).ravel()
            
            print("TN: {}".format(tn))
            print("TP: {}".format(tp))
            print("FN: {}".format(fn))
            print("FP: {}".format(fp))
            logging.info("TN: {}".format(tn))
            logging.info("TP: {}".format(tp))
            logging.info("FN: {}".format(fn))
            logging.info("FP: {}".format(fp))
        
            print("Test precision at {}: {}".format(score_k_val, precision))
            print("Test recall at {}: {}".format(score_k_val, recall))
            logging.info("Test precision at {}: {}".format(score_k_val, precision))
            logging.info("Test recall at {}: {}".format(score_k_val, recall))

        except Exception as e:
            log_msg = "MODEL BUILD ERROR: MODEL FAILED"
            print(e)
            logging.error(log_msg)
            logging.error(e)

    # PART 3: BASELINES

    pipeline.top_10_baseline(y_test, X_test, score_k_val)
    pipeline.top_ticket_baseline(y_test, X_test, score_k_val)
    pipeline.random_baseline(y_test, X_test, score_k_val)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()

    run(args.config)

