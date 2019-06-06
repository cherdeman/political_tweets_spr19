from utils.db_client import DBClient
from ETL.data_clean import DataClean
import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval

select_hash_dem = ["#bluewave2018", "#bluewave18", "#bluewave", "#democrats", "#resist", "#resistance", "#voteblue", 
"#Votethemout", "#WinBlue", "#YesWeCan", "#FlipItBlue"]

select_hash_rep = ["#trump", "#maga", "#gop", "#republican", "#trumptrain", "#kag", "#LeadRight", "#VoteRed", 
"#RedWave", "#VoteDemsOut"]

def run(table, chunk_size, strip_handles, rem_hashtags, to_table):
    db = DBClient()
    if "democrat" in table:
        dc = DataClean(select_hash_dem)
    else:
        dc = DataClean(select_hash_rep)

    chunk_size = int(chunk_size)
    query = "select * from {}".format(table)
    print(query)

    data = db.read_batch(query, chunk_size)
    data = pd.DataFrame(data)
    if "twitter140" in table:
        data.rename(index =str, columns = {0: 'label', 1: 'id', 2: 'date', 
        3: 'flag', 4: 'user', 5: 'tweet_text_raw'}, inplace=True)  
    else: 
        data.rename(index =str, columns = {0: 'tweet_id', 1: 'tweet_date', 2: 'tweet_text_raw', 
        3: 'user_id', 4: 'retweet_count', 5: 'favorite_count'}, inplace=True)    
    
    data['tweet_text_clean'] = data['tweet_text_raw'].apply(lambda x: dc.pre_process(x, strip_handles, rem_hashtags))
    data['len'] = data['tweet_text_clean'].apply(lambda x: len(x))
    data = data[data['len'] > 0]
    data.loc[data.label == 4, 'label'] = 1
    data['bigrams'] = data['tweet_text_clean'].apply(lambda x: dc.bigram(x, rem_hashtags))
    if "twitter140" in table:
        data['political'] = True
    else:
        data['political'] = data['bigrams'].apply(dc.political)
    
    # drop tweets that do not contain any political bigrams
    data_political = data[data['political']== True]

    # create leadership column (1 if leadership)
    if table in ("raw.senate", "raw.house"):
        data_political['leadership'] = 1
    else:
        data_political['leadership'] = 0

    data_political['topics'] = data_political['bigrams'].apply(dc.topics)
    topic_list = list(set([st for row in data_political['topics'] for st in row]))
    topic_list.sort()
    mlb = MultiLabelBinarizer()
    data_political = data_political.join(pd.DataFrame(mlb.fit_transform(data_political.pop('topics')),
                        columns=mlb.classes_,
                        index=data_political.index))

    data_political.drop(['bigrams', 'political', 'len'], inplace = True)
    file_path = "data/{}_clean.csv".format(table)
    data_political.to_csv(file_path, index = False)
    
    to_table_name = "staging.{}".format(to_table)
    drop_table = "DROP TABLE if exists {}".format(to_table_name)
    
    if "twitter140" in table:
        print("here")
        create_table = '''CREATE TABLE {} (label smallint, tweet_id varchar(20), tweet_date date, flag varchar(20), user_name varchar(30), tweet_text_raw text, tweet_text_clean text, leadership boolean, '''.format(to_table_name)
        for topic in topic_list:
            create_table += "{} boolean, ".format(topic)
        create_table = create_table[:-2] + ")"
    else:
        create_table = '''CREATE TABLE {} (tweet_id text, tweet_date date, tweet_text_raw text, user_id text, retweet_count int, favorite_count int, tweet_text_clean text, leadership boolean, '''.format(to_table_name)
        for topic in topic_list:
            create_table += "{} boolean, ".format(topic)
        create_table = create_table[:-2] + ")"
    
    db.write(["CREATE SCHEMA IF NOT EXISTS staging", drop_table, create_table])
    db.copy(file_path, "COPY {} FROM STDIN CSV HEADER".format(to_table_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table", help="name of table to clean")
    parser.add_argument("chunk_size", help="size of chunk to load data")
    parser.add_argument("--strip_handles", action = "store_true",  help="whether handles should be removed")
    parser.add_argument("--rem_hashtags", help="value 'all' if remove all hashtags and 'select' if remove selection hashtags")
    parser.add_argument("--to_table", help="name of table to write clean data")

    args = parser.parse_args()
    
    if args.to_table:
        to_table = args.to_table
    else:
        to_table = args.table.split('.')[1]

    run(args.table, args.chunk_size, args.strip_handles, args.rem_hashtags, to_table)
