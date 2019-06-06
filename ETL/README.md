# ETL ReadMe

This directory contains code to load sampled tweets into the database and make all transformations necessary to support the analysis. If you have not already, follow the "Database Setup" and "Twitter API Setup" steps at the root of the repository before following these steps.

## 1. Sample Tweets
The George Washington University "2018 U.S. Congressional Election Tweet Ids" project provided a number of CSVs that include lists of tweeet IDs related to the 2018 election. These lists are located in the `data/partisan_data` directory (not pushed due to filesize restrictions in Github). To sample tweets from the democratic and republic files, move to the root of the repository and run `./ETL/sample.sh`. You may edit the file to alter the number of tweets sampled from each input file and the output files the tweet samples should be written to.

## 2. Pull Sampled Tweets from Twitter API
The input file should be the output file of the previous step, i.e. a list of sampled Tweet IDs. The output file should be a file located in the  `data/tweets` directory with a CSV extension. 

From the root of the repository run the following command:
`python -m ETL.get_tweets <path to input file> <path to output file>`

You may need to run this step multiple times for each sample list.

Our convention was to name our output files in the following manner:
* senate tweets: "data/tweets/senate_tweets.csv"
* house tweets: "data/tweets/house_tweets.csv"
* democratic base tweets: "data/tweets/dem_tweets.csv" 
* democratic base tweets: "data/tweets/rep_tweets.csv"
         
If you choose a different convention, you will need to update the `tables` dictionary in `load.py` for step 3

## 3. Load Raw Tweets into Database
To load the raw tweets into the database, simply run `python -m ETL.load` from the root of the repository. Note that this step also requires that The party affiliations for House and Senate twitter accounts are located in files `data/lookups/house-accounts.csv` and `data/lookups/senate-accounts.csv`, respectively. The training sentiment140 data should be located in `data/training/train_twitter140.csv`.

## 4. Clean and Categorize Tweets by Topic

To clean the raw data and create the staging tables, run the `exec_data_clean.py` script, which uses the DataClean class and its associated methods from `data_clean.py`. The `exec_data_clean.py` script has the following command line arguments:
- `table`: the name of the raw database table to clean (should be raw.democrat, raw.republican, raw.senate, raw.house, or raw.train_twitter140)
- `chunk_size`: the number of rows in each chunk in the batch read from the database
- `--strip_handles` (optional): if this flag is present, handles will be removed during cleaning
- `--rem_hashtags` (optional): this flag should be followed by one of the two following values:
         - "all" if all hashtags should be removed
         - "select" if only the hashtags used to select the tweets should be removed (only relevant for raw.democrat and raw.republican)
- `--to_table` (optional): the name of the table to write the clean data, if this flag is not specified, the data will be written to staging.<name> where <name> comes from the raw table (eg. raw.democrat would be written to staging.democrat)

To clean the data as we did for analysis, run the following commands:

```
python3 -m ETL.exec_data_clean raw.democrat 100000 --strip_handles
python3 -m ETL.exec_data_clean raw.republican 100000 --strip_handles
python3 -m ETL.exec_data_clean raw.senate 100000 --strip_handles
python3 -m ETL.exec_data_clean raw.house 100000 --strip_handles
python3 -m ETL.exec_data_clean raw.train_twitter140 100000 --strip_handles
```
## 5. Create Master Table

Finally, create the master analysis table by running `python -m ETL.make_master` from the root of the repository.
