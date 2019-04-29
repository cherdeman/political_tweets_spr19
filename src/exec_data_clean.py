from db_client import DBClient
from analysis.data_clean import DataClean
import argparse
import pandas as pd

def run(table, chunk_size):
    db = DBClient()
    dc = DataClean()

    chunk_size = int(chunk_size)
    query = "select * from {}".format(table)
    print(query)

    data = db.read_batch(query, chunk_size)
    data = pd.DataFrame(data)
    data.rename(index =str, columns = {0: 'tweet_id', 1: 'tweet_date', 2: 'tweet_text_raw', 
    3: 'user_id', 4: 'retweet_count', 5: 'favorite_count'}, inplace=True)    
    
    data['tweet_text_clean'] = data['tweet_text_raw'].apply(lambda x: dc.pre_process(x))
    file_path = "../data/{}_clean.csv".format(table)
    data.to_csv(file_path, index = False)

    to_table_name = "staging.{}".format(table.split('.')[1])

    drop_table = "DROP TABLE if exists {}".format(to_table_name)
    create_table = '''CREATE TABLE {} (tweet_id text, tweet_date date, 
    tweet_text_raw text, user_id text, retweet_count int, favorite_count int, tweet_text_clean text)'''.format(to_table_name)

    db.write(["CREATE SCHEMA IF NOT EXISTS staging", drop_table, create_table])
    db.copy(file_path, "COPY {} FROM STDIN CSV HEADER".format(to_table_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table", help="name of table to clean")
    parser.add_argument("chunk_size", help="size of chunk to load data")
    args = parser.parse_args()

    run(args.table, args.chunk_size)
