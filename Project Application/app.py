import json

from flask import Flask, render_template, request, jsonify
import pandas as pd
import tweepy
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import scipy.stats as stats

app = Flask(__name__)

nltk.download('vader_lexicon')


@app.route("/")
def index():
   return render_template('index.html', meta_data_sentiment = 0, data_sentiment = 0, selected_items = 0)
  
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPv4iQEAAAAAyOqc1%2BBE%2B%2FP47s3U6Z5MymARK0I%3D8E2Jvl8oeKgr9JezE14QQosFZwmSAbfKitEXRCkROLupPKpll3'
client = tweepy.Client(bearer_token= bearer_token)
   
def get_tweets_stream(number_tweets, rule):
    filters_list = []
    class IDPrinter(tweepy.StreamingClient):
        def __init__(self, bearer_token):
            super().__init__(bearer_token)
            self.num_tweets = 0
        def on_tweet(self, tweet):
            record = tweet.text
            self.num_tweets += 1
        if self.num_tweets <= number_tweets:
            filters_list.append(record)
        else:
            self.disconnect()
            printer = IDPrinter(bearer_token)
    printer.add_rules(tweepy.StreamRule(rule))
    printer.get_rules()
    printer.filter()
    return filters_list
  
def get_sentiment_textblob(tweets_list):
    scores = []
    for tweet in tweets_list:
        sentiment = TextBlob(tweet).sentiment
        scores.append({'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity})
    frame = pd.DataFrame(scores)
    return frame
    
def get_sentiment_scores(tweets_list):
    scores = []
    sentiment = SentimentIntensityAnalyzer()
    for tweet in tweets_list:
        ss = sentiment.polarity_scores(tweet)
        scores.append(ss)
    frame = pd.DataFrame(scores)
    return frame
    
def clean_text(text):
    text = re.sub("RT @[\w]*:","",text)
    text = re.sub("(@[A-Za-z0-9_]+)","", text)
    text = re.sub("https?://[A-Za-z0-9./]*","",text)
    return text
    
def clean_tweets(tweets):
    return np.vectorize(clean_text)(tweets)
  
def get_all_sentiment_clean(tweets):
    clean_version = clean_tweets(tweets)
    vader_scores = get_sentiment_scores(clean_version)
    textblob_scores = get_sentiment_textblob(clean_version)
    return pd.concat([vader_scores, textblob_scores], axis=1)

def get_tweets_username(username, num_tweets):
    user = client.get_user(username = username).data
    if user == None:
        return "User does not exist"
    id_string = str(user.id)
    if num_tweets <= 0 or num_tweets > 3200:
        return "Invalid number of tweets"
    elif num_tweets <= 100:
        tweets = client.get_users_tweets(id = user.id, tweet_fields = ['created_at','referenced_tweets','in_reply_to_user_id','public_metrics'], max_results = num_tweets)
        if tweets.data == None:
            return "No tweets"
        tweets_text = [[get.text, get.data.get("referenced_tweets", [{"type": "solo"}])[0]["type"], get.data.get("in_reply_to_user_id", 0), get.data["public_metrics"]["retweet_count"], 
                    get.data["public_metrics"]["like_count"], get.data["public_metrics"]["reply_count"], get.data["created_at"]] for get in tweets.data]
        tweets_frame = pd.DataFrame(tweets_text, columns=['text', 'type', 'reply', 'retweets', 'likes', 'replies', 'time'])
    elif num_tweets <= 3200: 
        temp = []
        tweets = tweepy.Paginator(client.get_users_tweets, id = user.id, tweet_fields = ['created_at','referenced_tweets','in_reply_to_user_id','public_metrics'],
                                max_results=100).flatten(limit = num_tweets)
        for tweet in tweets:
            temp.append(tweet)
        if len(temp) == 0:
            return "No tweets"
        tweets_text = [[get.text, get.data.get("referenced_tweets", [{"type": "solo"}])[0]["type"], get.data.get("in_reply_to_user_id", 0), get.data["public_metrics"]["retweet_count"], 
                    get.data["public_metrics"]["like_count"], get.data["public_metrics"]["reply_count"], get.data["created_at"]] for get in temp]
        tweets_frame = pd.DataFrame(tweets_text, columns=['text', 'type', 'reply', 'retweets', 'likes', 'replies', 'time'])   

    condlist = [
        tweets_frame['reply'] == 0,
        tweets_frame['reply'] == id_string,
        tweets_frame['reply'] != id_string,
    ]
    choicelist = [
        0,
        0,
        1,
    ]
    tweets_frame['reply'] = np.select(condlist, choicelist)
    tweets_frame['author'] = username
    return tweets_frame

def get_tweets_query(query, num_tweets, hashtag = False):
    if hashtag == True:
        query = "#" + query
    else:
        query = '"' + query + '"'
    # By default, each query includes " -is:retweet lang:en"
    if num_tweets <= 0 or num_tweets > 6400:
        return "Invalid number of tweets"
    elif num_tweets <= 100:
        tweets = client.search_recent_tweets(query = query + " -is:retweet lang:en", tweet_fields = ['created_at','referenced_tweets','in_reply_to_user_id','public_metrics'], max_results = num_tweets)
        if tweets.data == None:
            return "No tweets"
        tweets_text = [[get.text, get.data.get("referenced_tweets", [{"type": "solo"}])[0]["type"], get.data.get("in_reply_to_user_id", 0), get.data["public_metrics"]["retweet_count"], 
                    get.data["public_metrics"]["like_count"], get.data["public_metrics"]["reply_count"], get.data["created_at"]] for get in tweets.data]
        tweets_frame = pd.DataFrame(tweets_text, columns=['text', 'type', 'reply', 'retweets', 'likes', 'replies', 'time'])
    elif num_tweets <= 6400: 
        temp = []
        tweets = tweepy.Paginator(client.search_recent_tweets, query = query + " -is:retweet lang:en", tweet_fields = ['created_at','referenced_tweets','in_reply_to_user_id','public_metrics'],
                                max_results=100).flatten(limit = num_tweets)
        for tweet in tweets:
            temp.append(tweet)
        if len(temp) == 0:
            return "No tweets"
        tweets_text = [[get.text, get.data.get("referenced_tweets", [{"type": "solo"}])[0]["type"], get.data.get("in_reply_to_user_id", 0), get.data["public_metrics"]["retweet_count"], 
                    get.data["public_metrics"]["like_count"], get.data["public_metrics"]["reply_count"], get.data["created_at"]] for get in temp]
        tweets_frame = pd.DataFrame(tweets_text, columns=['text', 'type', 'reply', 'retweets', 'likes', 'replies', 'time'])   
    tweets_frame["author"] = query
    return tweets_frame

def get_tweets_listusernames(usernames, num_tweets):
    if num_tweets < 0 or num_tweets > 3200:
        return "Invalid number of tweets"
    lists = []
    for username in usernames:
        result = get_tweets_username(username, num_tweets)
        if isinstance(result, pd.DataFrame):
            lists.append(result)
    return pd.concat(lists)
  
  
def compare_sentiment(sent1, sent2):
    res = {}
    for (columnName, data1),  (_, data2) in zip(sent1.iteritems(), sent2.iteritems()):
        res[columnName] = stats.ttest_ind(a = data1, b = data2, equal_var= False)
    return res

def proportion_sentiment2(compounds, cutoff = 0.1):
    if cutoff > 0.5:
        cutoff = 0.5
    if cutoff < 0.03:
        cutoff = 0.03
    res = [0,0,0]
    for value in compounds:
        if value >= cutoff:
            res[0] = res[0] + 1
        elif value >= -cutoff:
            res[1] = res[1] + 1
        else:
            res[2] = res[2] + 1
    return res

def proportion_sentiment(compounds, cutoffs = [-1, -0.1, 0.1, 1]):
    return list(pd.cut(compounds, cutoffs).value_counts(normalize=True, sort = False))


def add_sentiment_scores(df, column = "text"):
    temp = get_all_sentiment_clean(df[column])
    return pd.concat([df.reset_index(), temp], axis = 1)
   
def get_text_length_vectorize(text):
    text = re.sub("RT @[\w]*:","",str(text))
    text = re.sub("(@[A-Za-z0-9_]+)","", text)
    text = re.sub("https?://[A-Za-z0-9./]*","",text)
    return len(text)

def get_text_length(text):
    return np.vectorize(get_text_length_vectorize)(text)



def process(data_frame, author):
    decile = list(np.linspace(0, 1, num=11))
    twenty = list(np.linspace(-1, 1, num=21))
    # Negative
    if "time" in data_frame.columns:
        s_time = min(data_frame['time'])
        e_time = max(data_frame['time'])
    else:
        s_time = "Unknown"
        e_time = "Unknown"
    meta_data = {'start_time': s_time, 'end_time': e_time, 'count': data_frame.shape[0], 'author': author, 
                'neg': data_frame['neg'].mean(),
                'neu': data_frame['neu'].mean(),
                'pos': data_frame['pos'].mean(),
                'compound': data_frame['compound'].mean(),
                'polarity': data_frame['polarity'].mean(),
                'subjectivity': data_frame['subjectivity'].mean(),
                'length': data_frame['length'].mean()}
    first = pd.DataFrame(proportion_sentiment(data_frame['neg'], cutoffs = decile), columns = ['value'])
    first['quantile'] = decile[:-1]
    first['type'] = 'neg'
    # Neutral
    second = pd.DataFrame(proportion_sentiment(data_frame['neu'], cutoffs = decile), columns = ['value'])
    second['quantile'] = decile[:-1]
    second['type'] = 'neu'
    first = pd.concat([first, second])
    # Positive
    second = pd.DataFrame(proportion_sentiment(data_frame['pos'], cutoffs = decile), columns = ['value'])
    second['quantile'] = decile[:-1]
    second['type'] = 'pos'
    first = pd.concat([first, second])
    # Compound
    second = pd.DataFrame(proportion_sentiment(data_frame['compound'], cutoffs = twenty), columns = ['value'])
    second['quantile'] = twenty[:-1]
    second['type'] = 'compound'
    first = pd.concat([first, second])
    # Polarity 
    second = pd.DataFrame(proportion_sentiment(data_frame['polarity'], cutoffs = twenty), columns = ['value'])
    second['quantile'] = twenty[:-1]
    second['type'] = 'polarity'
    first = pd.concat([first, second])
    # Subjectivity 
    second = pd.DataFrame(proportion_sentiment(data_frame['polarity'], cutoffs = decile), columns = ['value'])
    second['quantile'] = decile[:-1]
    second['type'] = 'subjectivity'
    first = pd.concat([first, second])
    first['author'] = author
    return meta_data, first

@app.route("/get_data", methods=['POST'])
def get_data():
    items = [str(request.form['list1choice']), str(request.form['list5choice']), str(request.form['list3choice']), str(request.form['list4choice'])]
    options = ["Baseline","NLTKBaseline","query:abortion","query:America","query:California","query:CEO","query:China","query:climate change","query:climate warming",
	"query:conservatives","query:covid-19","query:Democrats","query:Donald Trump","query:Elon Musk","query:Florida","query:gas prices","query:global warming","query:guns","query:immigrants","query:inflation",     
	"query:Joe Biden","query:liberals","query:nuclear energy","query:oil price","query:president","query:Qatar","query:Republicans","query:Russia","query:San Francisco","query:solar energy","query:Tesla",
	"query:Texas","query:Toyota","query:Ukraine","user:AOC","user:AP","user:BarackObama","user:BBCBreaking","user:BBCWorld","user:BernieSanders","user:BillGates","user:BreitbartNews","user:business",
	"user:charliekirk11","user:chrislhayes","user:ClimateHuman","user:CNN","user:DeItaone","user:DonaldJTrumpJr","user:DougJBalloon","user:elonmusk","user:FoxNews","user:FT","user:GOP","user:GovRonDeSantis",
	"user:GretaThunberg","user:JoeBiden","user:KellyannePolls","user:KingJames","user:KyrieIrving","user:mattyglesias","user:MrBeast","user:Nate_Cohn","user:NateSilver538","user:NHC_Atlantic","user:Noahpinion",
	"user:nytimes","user:OANN","user:RBReich","user:realDonaldTrump","user:seanhannity","user:SenWarren","user:tedcruz","user:Twitter","user:unusual_whales","user:WSJ","user:YouTube"];
    
    list_meta_data = []
    list_results = []
    for i in range(4):
        item = items[i]
        if item in options:
            if item == "NLTKBaseline":
                meta_data, result = process(pd.read_csv("data_sep/NLTKcorpus.csv"), "NLTKBaseline")
            elif item == "Baseline":
                meta_data, result = process(pd.read_csv("data_sep/Baseline.csv"), "Baseline")
            elif item[:6] == "query:":
                meta_data, result = process(pd.read_csv("data_sep/ByQuery__" + item[6:] + ".csv"), item)
            elif item[:5] == "user:":
                meta_data, result = process(pd.read_csv("data_sep/ByAuthor_" + item[5:] + ".csv"), item)
        else:
            if item[:5] == "user:":
                tweets = get_tweets_username(item[5:], 1000)
                if not isinstance(tweets, pd.DataFrame):
                    error = "Error: No tweets for " + item
                    return render_template("index.html", meta_data_sentiment = error, data_sentiment = 0, selected_items = 0)
                else:
                    tweets = add_sentiment_scores(tweets)
                    tweets['length'] = get_text_length(tweets['text'])
                    meta_data, result = process(tweets, item)
            elif item[:6] == "query:":
                if item[6:] == "#":
                    tweets = get_tweets_query(item[7:], 1000, hashtag = True)
                else:
                    tweets = get_tweets_query(item[6:], 1000)
                if not isinstance(tweets, pd.DataFrame):
                    error = "Error: No tweets for " + item
                    return render_template("index.html", meta_data_sentiment = error, data_sentiment = 0, selected_items = 0)
                else:
                    tweets = add_sentiment_scores(tweets)
                    tweets['length'] = get_text_length(tweets['text'])
                    meta_data, result = process(tweets, item)
        list_meta_data.append(meta_data)
        list_results.append(result)
        
    list_results = pd.concat(list_results) 
    return render_template("index.html", meta_data_sentiment = list_meta_data, data_sentiment = list_results.reset_index().to_dict(), selected_items = items)

if __name__ == '__main__':
   app.run(debug=True)