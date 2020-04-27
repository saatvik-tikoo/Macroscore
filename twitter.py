import pandas as pd
import tweepy
import json
import numpy as np

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = '4754539741-PV9p5qifrGHvvpfIuOvC8mjojjbNRa6Qd47WJhn'
ACCESS_SECRET = 'qConSC7xKeGkMKrJC047zfppk4O6JjVvnISZgaT0GhHf2'
CONSUMER_KEY = 'Uy0h4bBb8ZpShTTY5s6kLhMFR'
CONSUMER_SECRET = 'yobtOpSrCdV9SxqiypahxYdDQsnV5DQWTIGwzaEBVzT0HFOquP'

# Setup tweepy to authenticate with Twitter credentials:
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
# auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

# Create the api to connect to twitter with your creadentials
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

# Get the metadata file
df = pd.read_csv('data/metadata.csv', usecols=['doi', 'title'])
df.dropna(inplace=True)
counts = np.zeros((1, df.shape[0]))[0]
for idx, row in df.iterrows():
    # Search for latest tweets based on dois
    tweets = tweepy.Cursor(api.search, q=row['doi'])
    output = []
    cnt = 0
    for status in tweets.items():
        output.append(status._json)
        cnt += 1
    print('Total results for ', row['doi'], ' is ', cnt)
    if cnt != 0:
        with open('data/tweets/{}.json'.format(str(row['doi']).replace('/', '__')), 'w') as outfile:
            json.dump(output, outfile)
        counts[idx] += cnt

    # Search for latest tweets based on dois
    tweets = tweepy.Cursor(api.search, q=row['title'])
    output = []
    cnt = 0
    for status in tweets.items():
        output.append(status._json)
        cnt += 1
    print('Total results for ', row['title'], ' is ', cnt)
    if cnt != 0:
        with open('data/tweets/{}.json'.format(str(row['title']).replace(' ', '__')), 'w') as outfile:
            json.dump(output, outfile)
        counts[idx] += cnt

df['tweet_count'] = counts
df.to_excel('data/tweet_counts.xlsx')
