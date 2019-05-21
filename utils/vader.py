from utils.db_client import DBClient
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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
        sid = SentimentIntensityAnalyzer()

        self.df['negative'] = None
        self.df['neutral'] = None
        self.df['positive'] = None
        self.df['compound'] = None

        for index, row in self.df.iterrows():
            tweet_text = row['tweet_text_raw']
            ss = sid.polarity_scores(tweet_text)
            self.df.iloc(index)['negative'] = ss['neg']
            self.df.iloc(index)['neutral'] = ss['neu']
            self.df.iloc(index)['positive'] = ss['pos']
            self.df.iloc(index)['compound'] = ss['compound']

    def outcome_by_group(self):
        