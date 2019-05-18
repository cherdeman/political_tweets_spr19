import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams
from gensim.utils import simple_preprocess
import re
import pandas as pd

class DataClean():
    def __init__(self, select_hash, cleaned_tweet_field = 'tweet_text_clean'):
        """[summary]
        
        :param table_name: [description]
        :type table_name: [type]
        :param cleaned_tweet_field: [description]
        :type cleaned_tweet_field: [type]
        """
        self.tweet_field = cleaned_tweet_field
        self.select_hash = select_hash
        self.topic_dict = self.create_topic_dict('ETL/final_topics.csv')

    def create_topic_dict(self,file_name):
        d = {}
        data = pd.read_csv(file_name)
        for i, row in data.iterrows():
            if row['Topic'] != 'x':
                d[row['Bigram']] = row['Topic']
        
        return d

    def rem_punctuation(self, tweet):
        # remove hashtag from punctuation because we wish to preserve
        punctuation = string.punctuation.replace('#', "")
        punctuation = punctuation.replace('@', "")
        punctuation += '’'
      
        translator = str.maketrans('','', punctuation)
        tweet = tweet.translate(translator)

        return tweet
        

    def rem_url(self, tweet):
        tweet = re.sub(r"http\S+", "", tweet)

        return tweet

    def tokenize(self, tweet, strip_handles):
        if strip_handles:
            tknzr = TweetTokenizer(preserve_case = False, strip_handles=True)
        else: 
            tknzr = TweetTokenizer(preserve_case = False)
        tweet = tknzr.tokenize(tweet)
        
        return tweet

    def get_wordnet_pos(self, tag):

        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def pos_tag(self, tokenized_tweet):
        tweet = nltk.pos_tag(tokenized_tweet)

        return tweet

    def lemmatize(self, tagged_tweet):
        lmtzr = WordNetLemmatizer()
        tweet = [lmtzr.lemmatize(i[0], self.get_wordnet_pos(i[1])) for i in tagged_tweet if len(i[0]) > 0]

        return tweet

    def remove_stop_words_len_one(self, tweet):
        stop_words = set(stopwords.words('english'))
        # amp is encoded for & 
        stop_words.add('amp')
        
        tweet_stop = ['like', 'follow', 'reply']

        tweet = [i for i in tweet if i not in stop_words and len(i) > 1] 
        
        return tweet

    def rem_hashtag(self, tweet, rem_hashtag):
        if rem_hashtag == "all":
            tweet = [i for i in tweet if not i.startswith("#")]
        elif rem_hashtag == "select":
            tweet = [i for i in tweet if i not in self.select_hash]
        
        return tweet

    def bigram(self, tweet, rem_hashtags):
        if rem_hashtags == "all":
            tweet = tweet + ["_".join(w) for w in ngrams(tweet, 2)]
        else:
            tweet = self.rem_hashtag(tweet, "all")
            tweet = tweet + ["_".join(w) for w in ngrams(tweet, 2)]

        return tweet

    def political(self, bigrams):
        if len(set(bigrams).intersection(self.topic_dict.keys())) > 0:
            return True
        else:
            return False

    def topics(self, bigrams):
        unique_topics = set([self.topic_dict.get(bigram, 0) for bigram in bigrams])

        return [x for x in unique_topics if x != 0]
        


    def pre_process(self, tweet, strip_handles, rem_hashtag):
        tweet = self.rem_punctuation(tweet)
        tweet = self.rem_url(tweet)
        tweet = self.tokenize(tweet, strip_handles)
        tweet = self.rem_hashtag(tweet, rem_hashtag)
        if len(tweet) > 0:
            tweet = self.pos_tag(tweet)
            tweet = self.lemmatize(tweet)
            tweet = self.remove_stop_words_len_one(tweet)
            return tweet
        else:
            return []

    
        


    