from utils.db_client import DBClient
from ETL.data_clean import DataClean
import argparse
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

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
    data.rename(index =str, columns = {0: 'tweet_id', 1: 'tweet_date', 2: 'tweet_text_raw', 
    3: 'user_id', 4: 'retweet_count', 5: 'favorite_count'}, inplace=True)    
    
    data['tweet_text_clean'] = data['tweet_text_raw'].apply(lambda x: dc.pre_process(x, strip_handles, rem_hashtags))
    data['bigrams'] = data['tweet_text_clean'].apply(lambda x: dc.bigram(x, rem_hashtags))
    data['political'] = data['bigrams'].apply(dc.political)
    
    # drop tweets that do not contain any political bigrams
    data_political = data[data['political']== True]

    # create leadership column (1 if leadership)
    if table in ("raw.senate", "raw.house"):
        data_political['leadership'] = 0
    else:
        data_political['leadership'] = 1

    # create one hot encoded columns for topics

    data_political['topics'] = data_political['bigrams'].apply(dc.topics)
    mlb = MultiLabelBinarizer()
    data_political = data_political.join(pd.DataFrame(mlb.fit_transform(data_political.pop('topics')),
                          columns=mlb.classes_,
                          index=data_political.index))

    data_political.drop(['bigrams', 'political'], axis=1, inplace = True)
    file_path = "data/{}_clean.csv".format(table)
    data_political.to_csv(file_path, index = False)

    #to_table_name = "staging.{}".format(table.split('.')[1])
    to_table_name = "staging.{}".format(to_table)

    drop_table = "DROP TABLE if exists {}".format(to_table_name)
    create_table = '''CREATE TABLE {} (tweet_id text, tweet_date date, 
    tweet_text_raw text, user_id text, retweet_count int, favorite_count int, tweet_text_clean text,
    leadership boolean, budget boolean, civil_rights boolean, courts boolean, criminal_justice boolean, 
    drugs boolean, econ_inequality boolean, econ_jobs boolean, education boolean, 
    environment boolean, family boolean, foreign_policy boolean, governance boolean,
    guns boolean, health boolean, immigration boolean, military boolean, 
    public_safety boolean, puerto_rico boolean, race boolean, rural boolean, russia boolean,
    sexual_assault boolean, shutdown boolean, social_security boolean, taxes boolean,
    technology boolean, women_rights boolean)'''.format(to_table_name)

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
