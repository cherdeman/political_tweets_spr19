# script to load tweet data files into database

import sys
sys.path.append("..")
from utils import db_client

tables = {"senate":"senate_tweets.csv",
          "house":"house_tweets.csv",
          "democrat":"dem_tweets.csv",
          "republican":"rep_tweets.csv" }

####### QUERIES ######

create_schema = "create schema if not exists raw"

drop = "drop table if exists {};"

create_table = """
               create table {} (
                  tweet_id varchar(20) NOT NULL,
                  tweet_date date,
                  tweet_text text,
                  user_id varchar(20) NOT NULL,
                  retweet_count int,
                  favorite_count int
                  );
               """

copy = """
       copy {}(tweet_id, tweet_date, tweet_text, user_id, retweet_count, favorite_count) 
       from STDIN delimiter ',' CSV HEADER;
       """


def main():
    db = db_client.DBClient()

    print(f"Creating raw schema if needed...")
    db.write([create_schema])

    for table, file in tables.items():
      full_table_name = "raw." + table

      print(F"Dropping and recreating table {full_table_name}")
      db.write([drop.format(full_table_name), create_table.format(full_table_name)])
      #db.write(create_table.format(full_table_name))

      print(f"Copying from {file} into {full_table_name}")
      csv_path = f"../../data/tweets/{file}"
      sql = copy.format(full_table_name)
      db.copy(csv_path, sql)


if __name__ == "__main__":
    main()




