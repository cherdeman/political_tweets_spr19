# queries to load tweet data files into database

tables = {"senate":["tweets","senate_tweets.csv"],
          "house": ["tweets","house_tweets.csv"],
          "democrat":["tweets","dem_tweets.csv"],
          "republican":["tweets","rep_tweets.csv"],
          "house_accounts": ["lookups","house-accounts.csv"],
          "senate_accounts": ["lookups","senate-accounts.csv"]}

####### QUERIES ######

create_schema = "create schema if not exists raw"

drop = "drop table if exists {};"

create_table = "create table {} ({});"

tweet_columns = """
                  tweet_id varchar(20) NOT NULL,
                  tweet_date date,
                  tweet_text text,
                  user_id varchar(20) NOT NULL,
                  retweet_count int,
                  favorite_count int
                """

accounts_columns = """
                    token varchar(20),
                    user_id varchar(20),
                    link varchar(50),
                    party_affiliation varchar(5),
                    primary key (user_id)
                   """

tweet_col_names = "tweet_id, tweet_date, tweet_text, user_id, retweet_count, favorite_count"
account_col_names = "token, user_id, link"#, party_affiliation"

copy = """
       copy {}({}) 
       from STDIN delimiter ',' CSV HEADER;
       """



