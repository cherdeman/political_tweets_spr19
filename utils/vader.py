from utils.db_client import DBClient
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VADER_Analyzer:
    def __init__(self, topic):
        self.topic = topic
        self.query = """
                    SELECT tweet_id, tweet_date, tweet_text_raw, user_id, democrat, leadership
                    from staging.master
                    where {} = true
                    """
        self.df = None
        self.results = None
        self.choices = ['dem-lead', 'dem-base', 'rep-lead', 'rep-base']
        self.palette = {"dem-base": "#a8b2ff","dem-lead": "#0015bc", "rep-base": "#ff9d9d", "rep-lead": "#ff0000", "": "gray"}


    def get_data(self):
        column_names = {0: "tweet_id", 
                        1: "tweet_date",
                        2: "tweet_text_raw",
                        3: "user_id",
                        4: "democrat", 
                        5: "leadership"
                        }
        
        db = DBClient(secrets_path = '../configs/db_secrets.json')
        q = self.query
        self.df = pd.DataFrame(db.read(q.format(self.topic))).rename(columns=column_names)


    def run_vader(self):
        sid = SentimentIntensityAnalyzer()

        self.df['negative'] = None
        self.df['neutral'] = None
        self.df['positive'] = None
        self.df['compound'] = None

        for index, row in self.df.iterrows():
            tweet_text = row['tweet_text_raw']
            ss = sid.polarity_scores(tweet_text)
            self.df.at[index, 'negative'] = ss['neg']
            self.df.at[index, 'neutral'] = ss['neu']
            self.df.at[index, 'positive'] = ss['pos']
            self.df.at[index, 'compound'] = ss['compound']

        self.df['negative'] = self.df['negative'].astype(float)
        self.df['neutral'] = self.df['neutral'].astype(float)
        self.df['positive'] = self.df['positive'].astype(float)
        self.df['compound'] = self.df['compound'].astype(float)

    def outcome_by_group(self):
        self.results = self.df.groupby(['democrat','leadership'], 
            as_index=False).agg(
                              {'negative':['mean','std', 'count'],
                               'neutral':['mean','std'],
                               'positive':['mean','std'],
                               'compound':['mean','std']
                              }
                              )

    def results_for_plotting(self):
        self.df['group'] = np.where((self.df['democrat']==True) & (self.df['leadership']==True), 'dem-lead', 
            np.where((self.df['democrat']==True) & (self.df['leadership']==False), 'dem-base',
            np.where((self.df['democrat']==False) & (self.df['leadership']==True), 'rep-lead', 
            np.where((self.df['democrat']==False) & (self.df['leadership']==False), 'rep-base', ""))))

        self.plotting = pd.melt(self.df[self.df['group'] != ""], id_vars = ['democrat', 'leadership', 'group', 'tweet_id', 'tweet_date', 'tweet_text_raw', 'user_id'])
  
    def plot(self, neg_pos_only = True):
        if neg_pos_only:
            data = self.plotting[self.plotting['variable'].isin(['negative', 'positive'])]
        else:
            data = self.plotting

        return sns.barplot(x='variable', y='value', hue='group', data=data, palette=self.palette)
