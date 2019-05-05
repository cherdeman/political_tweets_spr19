import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams
from gensim.utils import simple_preprocess
import re



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
        tweet = [i for i in tweet if i not in stop_words and len(i) > 1] 
        
        return tweet

    def rem_hashtag(self, tweet, rem_hashtag):
        if rem_hashtag == "all":
            tweet = [i for i in tweet if not i.startswith("#")]
        elif rem_hashtag == "select":
            tweet = [i for i in tweet if i not in self.select_hash]
        
        return tweet

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
        


    