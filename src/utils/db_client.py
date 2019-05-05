import json
import csv
import psycopg2
import os
import atexit
import logging
import sys

logger = logging.getLogger('ptlog')
sh = logging.StreamHandler(sys.stdout)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

class DBClient():

    def __init__(self, if_local_connection = True, secrets_path = '../../configs/db_secrets.json', 
                 schema_name = None, db_name = "politicaltweets"):
        """
        Class for maintaining the database client object, with attributes and 
        methods required for connecting to a DB, copying files into the DB,
        and executing fetch, DDL, DML queries. 
        
        :param if_local_connection: Flag which denotes if client should 
                                    connect to a locally running DB server, 
                                    defaults to True
        :param if_local_connection: bool, optional
        :param secrets_path: Path of the secrets file containing the 
                             credentials and variables required to connect to
                             the DB. Overriden by if_local_connection,
                             defaults to None
        :param secrets_path: str, optional
        :param schema_name: Name of the schema to set to on the database, 
                            defaults to None
        :param schema_name: str, optional
        :param db_name: Name of the database to which the client should connect, 
                        defaults to "visionzero"
        :param db_name: str, optional
        """

        try:
            #secrets_file =  secrets_path
            with open(secrets_path) as f:
                env = json.load(f)
            self.DB_USER = env['DB_USER']
            self.DB_PASSWORD = env['DB_PASSWORD']
            self.DB_HOST = env['DB_HOST']
            self.DB_PORT = env['DB_PORT']
            self.DB_NAME = env['DB_NAME']

            conn = psycopg2.connect(database=self.DB_NAME, user=self.DB_USER,
                                    password=self.DB_PASSWORD, host=self.DB_HOST, 
                                    port=self.DB_PORT)
        except Exception as e:
            print("Error in connecting to database " + db_name)
            print(e)
            
        finally:
            print("Connected to political tweets DB")
            self.conn = conn
            self.cur = self.conn.cursor()

        try:
            if schema_name is not None:
                conn.cursor().execute("SET SCHEMA '{}';".format(schema_name))
        except Exception as e:
            print("Error in setting schema to " + schema_name)
            print(e)

    def write(self, statements, values=None):
        """
        Execute statements, close the cursor on exit (write-only).
        
        :param statements: SQL statement to execute
        :type statements: str #TODO: @amstern, please enter description for below values
        :param values: [description], defaults to None
        :param values: [type], optional
        """

        with self.conn.cursor() as cur:
            for statement in statements:
                try:
                    cur.execute(statement)
                except Exception as e:
                    print("Exception in writing")
                    print(e)
                    self.conn.rollback()
                    logger.info("rollback")
                finally:
                    self.conn.commit()
                    logger.info("db write committed")

    def read(self, statement, args=None):
        """
        Execute a data query statement, returns all rows using fetchall.
        Note that since we're using fetchall, we'll be returning all the
        rows returned by the query. If you'd prefer to fetch rows in 
        batches, please refer the read_batch command. 
        
        :param statement: SQL query to execute
        :type statement: str #TODO: @amstern, please enter description for below values
        :param args: [description], defaults to None
        :param args: [type], optional
        :return: A tuple of tuples containing the rows from the fetch query
        :rtype: Tuple of tuples
        """

        l = []
        with self.conn.cursor() as cur:
            if not args:
                cur.execute(statement)
            else:
                cur.execute(statement, args)
            l = cur.fetchall()
        return l

    def copy(self, csv_path, statement, args=None):
        """
        Execute copy statement.
        
        :param csv_path: Path of the CSV file to be copied into the DB
        :type csv_path: str
        :param statement: SQL statement to execute the copy commad
        :type statement: str #TODO: @amstern, please enter description for below values
        :param args: [description], defaults to None
        :param args: [type], optional
        """

        with open(csv_path, 'r') as f:
            #cur = self.conn.cursor()
            try:
                self.cur.copy_expert(sql=statement, file=f)
            except Exception as e:
                print('Entering Exception')
                print(e)
                self.conn.rollback()
            finally:
                self.conn.commit()            

    def exit(self):
        """
        Method to exit the database
        
        """

        self.conn.close()
        logger.info("Connection closed.")
