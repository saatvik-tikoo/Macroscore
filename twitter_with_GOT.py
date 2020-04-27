import json
import sys
import tweepy
import pandas as pd
import numpy as np

CONSUMER_KEY = 'Uy0h4bBb8ZpShTTY5s6kLhMFR'
CONSUMER_SECRET = 'yobtOpSrCdV9SxqiypahxYdDQsnV5DQWTIGwzaEBVzT0HFOquP'

auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

if not api:
    print("Can't Authenticate")
    sys.exit(-1)

df = pd.read_csv('data/metadata.csv', usecols=['doi', 'title'])
df.dropna(inplace=True)


# Search for latest tweets based on dois
def get_from_dois():

    for idx, row in df.iterrows():
        output = []
        searchQuery = row['doi']

        maxTweets = 1000
        tweetsPerQry = 100
        sinceId = None
        max_id = -1
        tweetCount = 0

        while tweetCount < maxTweets:
            try:
                if max_id <= 0:
                    if not sinceId:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, since_id=sinceId)
                else:
                    if not sinceId:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), since_id=sinceId)

                if not new_tweets:
                    print("No more tweets found")
                    break
                else:
                    print("Got tweets for {} with doi {}".format(idx, row['doi']))

                for tweet in new_tweets:
                    output.append(tweet._json)

                tweetCount += len(new_tweets)
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

        if len(output) > 0:
            with open('data/tweets/{}.json'.format(str(row['doi']).replace('/', '__')), 'w') as outfile:
                json.dump(output, outfile)


# Search for latest tweets based on title
def get_from_titles():

    for idx, row in df.iterrows():
        output = []
        searchQuery = row['title']

        maxTweets = 1000
        tweetsPerQry = 100
        sinceId = None
        max_id = -1
        tweetCount = 0

        while tweetCount < maxTweets:
            try:
                if max_id <= 0:
                    if not sinceId:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, since_id=sinceId)
                else:
                    if not sinceId:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), since_id=sinceId)

                if not new_tweets:
                    print("No more tweets found")
                    break
                else:
                    print("Got tweets for {} with title {}".format(idx, row['title']))

                for tweet in new_tweets:
                    output.append(tweet._json)

                tweetCount += len(new_tweets)
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

        if len(output) > 0:
            with open('data/tweets/{}.json'.format(str(row['title']).replace(' ', '__').replace('/', '__')), 'w') as outfile:
                json.dump(output, outfile)


if __name__ == '__main__':
    get_from_titles()
    get_from_dois()
