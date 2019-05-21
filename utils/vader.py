from utils.db_client import DBClient
import pandas as pd


class VADER_analyzer:
    def __init__(self, topic):
        self.topic = topic
        self.query = """
                    SELECT tweet_id, tweet_date, tweet_text_raw, user_id, democrat, leadership
                    from staging.master
                    where {} = true
                    """
        self.df = None


    def get_data(self):
        db = DBClient()
        q = self.query
        self.df = pd.DataFrame(db.read(q.format(self.topic)))


    def run_vader(self):
        pass

    def outcome_by_group(self):
        pass