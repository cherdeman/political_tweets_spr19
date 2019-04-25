import json
import psycopg2

def connect():
    with open('../configs/db_config.json') as f:
        config = json.load(f)
        host=config["host"]
        username=config["user"]
        database=config["db"]
        password=config["pass"]
        port=config["port"]
        
        return psycopg2.connect(database=database, user=username,
                                password=password, host=host, port=port)


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

####### FUNCTIONS #######

def create_raw_schema(cur):
    print(f"Creating raw schema if needed...")
    cur.execute(create_schema)

def drop_and_recreate_table(cur, full_table_name):
    print(F"Dropping and recreating table {full_table_name}")
    cur.execute(drop.format(full_table_name))
    cur.execute(create_table.format(full_table_name))

def copy_file_to_table(cur, file, full_table_name):
    print(f"Copying from {file} into {full_table_name}")
    with open('../data/tweets/{}'.format(file), 'r') as t:
        next(t)
        cur.copy_expert(copy.format(full_table_name), file=t)
        t.close()


def main():
    conn = connect()
    cur = conn.cursor()

    create_raw_schema(cur)
    conn.commit()

    for table, file in tables.items():
        full_table_name = "raw." + table

        drop_and_recreate_table(cur, full_table_name)
        conn.commit()

        copy_file_to_table(cur, file, full_table_name)
        conn.commit()

    print("Done!")


if __name__ == "__main__":
    main()




