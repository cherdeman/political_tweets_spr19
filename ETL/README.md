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
To load the raw tweets into the database, simply run `python -m ETL.load` from the root of the repository. Note that this step also requires that The party affiliations for House and Senate twitter accounts are located in files `data/lookups/house-accounts.csv` and `data/lookups/senate-accounts.csv`, respectively.

[ALENA TO DO: Where does thte sentiment 140 data need to live to run load.py]

## 4. Clean and Categorize Tweets by Topic

[ALENA TO DO: Cleaning steps]

Finally, create the master analysis table by running `python -m ETL.make_master` from the root of the repository.
