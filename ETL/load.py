# queries and functions to load tweet data files into database
from utils.db_client import DBClient

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
                    party_affiliation varchar(1),
                    primary key (user_id)
                   """

tweet_col_names = "tweet_id, tweet_date, tweet_text, user_id, retweet_count, favorite_count"
account_col_names = "token, user_id, link, party_affiliation"

copy = """
       copy {}({}) 
       from STDIN delimiter ',' CSV HEADER;
       """

### Loading function

def prepare_statements(file_directory, full_table_name):
    if file_directory == "tweets":
        create_statement = create_table.format(full_table_name, tweet_columns)
        copy_statement = copy.format(full_table_name, tweet_col_names)
    else:
        create_statement = create_table.format(full_table_name, accounts_columns)
        copy_statement = copy.format(full_table_name, account_col_names)

    drop_statement = drop.format(full_table_name)

    return create_statement, copy_statement, drop_statement

def load(db):

    print(f"Creating raw schema if needed...")
    db.write([create_schema])

    for table, file_list in tables.items():
        filedir = file_list[0]
        filename = file_list[1]
        full_table_name = "raw." + table

        create_statement, copy_statement, drop_statement = prepare_statements(filedir, full_table_name)        

        print(f"Dropping and recreating table {full_table_name}")
        db.write([drop_statement, create_statement])

        print(f"Copying from {filename} into {full_table_name}")
        csv_path = f"./data/{filedir}/{filename}"
        db.copy(csv_path, copy_statement)

    db.exit()

if __name__ == "__main__":
  db = DBClient()
  load(db)



