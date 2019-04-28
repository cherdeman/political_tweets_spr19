import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams
from gensim.utils import simple_preprocess



class DataClean():
    def __init__(self, table_name, cleaned_tweet_field = 'tweet_text_clean'):
        """[summary]
        
        :param table_name: [description]
        :type table_name: [type]
        :param cleaned_tweet_field: [description]
        :type cleaned_tweet_field: [type]
        """
        self.table_name = table_name
        self.tweet_field = cleaned_tweet_field

    # how to deal w/ URL? - remove!
    # convert everything to lower case? - yes!
    # remove emoji
    # 

    def rem_punctuation(self, chunk):
        tweet_text = self.tweet_field
        # remove hashtag from punctuation because we wish to preserve
        punctuation = string.punctuation.replace('#', "")
        punctuation = punctuation.replace('@', "")
        punctuation += '’'
        print(punctuation)
        translator = str.maketrans('','', punctuation)
        chunk[tweet_text] = chunk[tweet_text].str.translate(translator)

        return chunk 

    def tokenize(self, chunk):
        # do we want to remove handles?
        tknzr = TweetTokenizer(preserve_case = False)
        chunk[self.tweet_field] = chunk[self.tweet_field].apply(lambda x: tknzer.tokenize(x)))
        
        return chunk

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

    def lemmatize(self, chunk):
        lmtzr = WordNetLemmatizer()
        chunk[self.tweet_field] = chunk[self.tweet_field].apply(lambda x: nltk.pos_tag(x))
        chunk[self.tweet_field] = chunk[self.tweet_field].apply(lambda x: [lmtzr.lemmatize(i[0], 
        self.get_wordnet_pos(i[1])) for i in x if len(i[0]) > 0])

        return chunk

    def remove_stop_words(self, chunk):
        stop_words = set(stopwords.words('english'))
        chunk[self.tweet_field] = chunk[self.tweet_field].apply(lambda x: [i for i in x if i not in stop_words] )
        
        return chunk
