This repository contains the files used to complete all of the modeling for this project, which can be divided into three categories: 1) Baseline Models, 2) VADER, 3) LSTM.

## Baseline Models ##

This includes the following files:
- `sentiment_analysis.py`: contains the Pipeline class and associated methods
- `exec_sentiment_analysis.py`: execution script for modeling which uses Pipeline class to conduct gridsearch with cross-validation and then calculates validation and testing metrics. This file takes the path to to the configuration file as a command line argument.
- `sa_config.json`: configuration file where we specify parameters for model training, including the validation and testing fractions, the classifier and hyperparameter grid for gridsearch, and information for quering data from the database.

To execute the baseline model training, run the following from the command line in the root directory of this repository:
`python -m analysis.exec_sentiment_analysis analysis/sa_config.json`

The file `sa_config.json` contains the configuration that we used for model training in our analysis. It contains the following parameters:
- `iteration_name`: the name of the training iteration (identifies saved logs and trained model object)
- `feature_table_name`: name of database table containing training data
- `reqd_cols`: required columns for analysis (features and label)
- `test_frac`: fraction of data to be used for testing, must be between 0 and 1
- `val_frac`: fraction of data to be used for validation, must be between 0 and 1 (and test_frac and val_frac should not sum to more than 1. The remaining data will be used for training)
- `run_type`: "build" if training a model and "load" if loading a trained model to classify new data
- `grid`: classifier and hyperparameter grid for gridsearch
- `model_obj_path`: if run_type is load, should be path to saved model object
- `scoring`: scoring type for grid search optimization, must be one of [sklearn's scoring parameters](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

The `sentiment_analysis.py` and `exec_sentiment_analysis.py` files adapt code that Alena had previously written (with Hye and Vishwanath) for a Winter 2019 course. These files retain the same basic structure of the previous work, but implement several changes as follows:

*sentiment_analysis.py*
- The constructor is adapted from the previous code, namely stripping out unneeded attributes and altering how the scoring function for gridsearch is passed into the constructor.
- The `gen_pred_probs()`, `train_grid()`, and `load_model_obj()` methods are unchanged from the previous code and the `gen_preds()` method has been written new for this assignment.

*exec_sentiment_analysis.py*
- The general structure of the execution script is the same from the previous code (eg. first read in data, then split into X and y, etc.) and the command line argument code and logging code is adapted from the original. 
- The execution script was adapted to accommodate a train/validation/test split (the original only had train/test), with the resulting train/validation/test split code and evaluation code written new for this assignment.
- The code to pre-process the data from the database, create the TF-IDF features, and convert the resulting data into dense matrices for modeling was written new for this assignment. 

## VADER ##

The Vader unsupervised sentiment analysis was conducted in the `Vader Prelim.ipynb` notebook. To run this code, open the notebook and run the cells.

## LSTM ##

The LSTM supervised sentiment analysis was conducted in the `Final LSTM.ipynb` notebook. This notebook includes both the code to train the LSTM model on the sentiment140 training data, the code to apply the trained LSTM model to classify the sentiment the political Twitter data, and the code to analyze the political sentiment results. To run this code, open the notebook and run the cells.
