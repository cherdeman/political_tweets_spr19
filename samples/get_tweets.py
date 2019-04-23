import time
import sys
import tweepy
import json
import pandas as pd
import csv


def get_credentials(secrets_file = '../configs/secrets.json'):
    with open(secrets_file) as f:
        consumer_info = json.load(f)

    consumer_key = consumer_info["api_key"]
    consumer_secret = consumer_info["api_secret_key"]
    access_token = consumer_info["access_token"]
    access_token_secret = consumer_info["access_token_secret"]

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth)

def get_tweets(api, input_file, output_file):
    with open(output_file,'w') as f:
        writer = csv.writer(f, delimiter= ',')
        writer.writerow(['id', 'created_at', 'text', 'user_id', 'retweets', 'favorites'])
        
        reader = pd.read_csv(input_file, sep = '\n', header=None, names=['id'], chunksize = 100)
        for chunk in reader:
            try:
                tweets = api.statuses_lookup(list(chunk['id']), include_entities=True, trim_user=True, tweet_mode='extended')
            except tweepy.RateLimitError:
                print("You've been LIMITED, give it 15.")
                time.sleep(15 * 60)
                print("You're starting up again!")
                tweets = api.statuses_lookup(list(chunk['id']), include_entities=True, trim_user=True, tweet_mode='extended')
            except tweepy.TweepError as e:
                if 'Failed to send request:' in e.reason:
                    print("Time out error caught.")
                    time.sleep(180)
                    continue
            for status in tweets:
                if not hasattr(status, 'retweeted_status'):
                    row = [status.id_str, status.created_at, status.full_text, status.user.id_str, 
                       status.retweet_count, status.favorite_count]
                    writer.writerow(row)

def main():
    print("The input file is: ", sys.argv[1])
    print("The output file is: ", sys.argv[2])
    api = get_credentials()
    get_tweets(api, sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()