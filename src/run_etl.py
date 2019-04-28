import ETL.load as l
from db_client import DBClient 

### Loading function

def prepare_statements(file_directory, full_table_name):
    if file_directory == "tweets":
        create_statement = l.create_table.format(full_table_name, l.tweet_columns)
        copy_statement = l.copy.format(full_table_name, l.tweet_col_names)
    else:
        create_statement = l.create_table.format(full_table_name, l.accounts_columns)
        copy_statement = l.copy.format(full_table_name, l.account_col_names)

    drop_statement = l.drop.format(full_table_name)

    return create_statement, copy_statement, drop_statement

def load(db):

    print(f"Creating raw schema if needed...")
    db.write([l.create_schema])

    for table, file_list in l.tables.items():
        filedir = file_list[0]
        filename = file_list[1]
        full_table_name = "raw." + table

        create_statement, copy_statement, drop_statement = prepare_statements(filedir, full_table_name)        

        print(f"Dropping and recreating table {full_table_name}")
        db.write([drop_statement, create_statement])

        print(f"Copying from {filename} into {full_table_name}")
        csv_path = f"../data/{filedir}/{filename}"
        db.copy(csv_path, copy_statement)

    db.exit()


if __name__ == "__main__":
	db = DBClient()
	load(db)