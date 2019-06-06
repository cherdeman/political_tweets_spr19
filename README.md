# Political Tweets Spring 2019
Project repository for Spring 2019 Advanced Machine Learning Project.
Alena Stern & Claire Herdeman

# Project Description
Republican and Democratic Party Leadership are obliged to both set the agenda and political priorities for the party and respond to the concerns of their respective bases. Based on this, we believe that there may be differences in both what policy topics are discussed within each party and leadership/base group, and how they talk about those issues.

To explore this question further, analyzed tweets from the 2018 Midterm election cycle. We first defined policy topic areas based on frequently occuring bigrams, the analyzed the sentiment of tweets in each policy area and by party and leadership/base grouping using 4 methods:

* VADER Modeling (unsupervised)
* Naive Bayes (baseline - supervised)
* Logistic Regression (baseline - supervised)
* LSTM Neural Network Model (supervised)

# Running the Code

## Project Environment
This project was developed in Python 3.6. Package requirements can be found in the `requirements.txt` file and installed (into the virtual environment of your choice) with the command `pip install requirements.txt`. Note that the project also requires Anaconda to work with the Jupyter Notebooks.

### Database Setup
All of our raw and cleaned twitter data are located in a Postgres database hosted on AWS's Relational Database Service (RDS). We recommend setting up similar infrastucture to run our code with minimal refactoring. 

To connect to the database, create a file called `db_secrets.json` in the `config` directory. It should be formatted in the following manner (with values replaced as needed):

```
{"DB_HOST":"<your-db-host-name>",  
 "DB_USER":"<your-db-user>",  
 "DB_PASSWORD":"<your-db-password>",  
 "DB_PORT":5432,  
 "DB_NAME":"<your-db-name>"  
}
```

### Twitter API Setup
We used the Twitter API to gather sampled tweets from the 2018 election. In order to run a similar data gathering process, you must have twitter developer account and get API credentials. You can find instructions to complete this process [here](https://developer.twitter.com/). Then, create a file called `secrets.json` in the `config` directory. It should be formatted in the following manner (with values replaced as needed):

```
{"api_key":"<your-twitter-api-key>",
 "api_secret_key":"<your-twitter-secret-key>",
 "access_token":"<your-twitter-access-token>",
 "access_token_secret":"<your-twitter-access-token-secret>"}
```

### Repository Structure
This repository is structured as followed:
- `ETL/`: contains files for gathering data from Twitter API, loading data into database, identifying political bigrams and topics, and cleaning data for analysis.
- `analysis/`: contains files for modeling including baseline models, LSTM, and VADER. 
- `data/`: contains raw data files used for this project.
- `utils/`: contains a database connection utility and a vader utility.

### Code Adapted from Previous Work
Three files in this repository containcode that wasn't written new for this project: `analysis/sentiment_analysis.py`, `analysis/exec_sentiment_analysis.py`, and `utils/db_client.py`. The `utils/db_client.py` is largely unchanged for this assignment, though some small changes were made for this project. The two analysis files were changed far more singificantly for this assignment. The README in the analysis folder discusses in detail what code in these two files was repurposed for this assignment versus written new.

# Dataset Citations
Wrubel, Laura; Littman, Justin; Kerchner, Dan, 2019, "2018 U.S. Congressional Election Tweet Ids", https://doi.org/10.7910/DVN/AEZPLU, Harvard Dataverse, V1, UNF:6:gxmogmaacqF8Mu3nvM793w== [fileUNF]

* Sampled 100000 from each dem file (4)
* Sampled 50000 from each rep file (11)
* All from Senate file 200k
* Half of House 800k

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/). 

[Sentiment 140](http://www.sentiment140.com/)
