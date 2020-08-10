"""
Author @ Mihir_Srivastava
Dated - 17-07-2020
File - Sentiment_analysis_of_tweets_NB_scratch
Aim - To do the sentiment analysis of tweets from the twitter_samples DataSet available in the nltk library and classify
them as positive or negative tweets using Naive Bayes algorithm from scratch.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Collect positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Append positive and negative tweets to get all the tweets
all_tweets = positive_tweets + negative_tweets

# Split the data into two pieces, one for training and one for testing (validation set)
test_pos = positive_tweets[4000:]
train_pos = positive_tweets[:4000]
test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


# A function to visualize the classes
def visualize(all_positive_tweets, all_negative_tweets):
    # Declare a figure with a custom size
    fig = plt.figure(figsize=(5, 5))

    # labels for the two classes
    labels = 'Positives', 'Negative'

    # Sizes for each slide
    sizes = [len(all_positive_tweets), len(all_negative_tweets)]

    # Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Display the chart
    plt.show()


# A function to print a sample tweet from each class
def sample_tweets():
    print('\033[92m' + positive_tweets[random.randint(0, 5000)])
    print('\033[91m' + negative_tweets[random.randint(0, 5000)])


# A function to preprocess a tweet before feeding to our ML model
def process_tweet(tweet):
    # remove stock market tickers like $GE
    tweet2 = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet2 = re.sub(r'^RT[\s]+', '', tweet2)

    # remove hyperlinks
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)

    # remove hashtags (only removing the hash # sign from the word)
    tweet2 = re.sub(r'#', '', tweet2)

    # Create Tweet Tokenizer object
    token = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)

    # Tokenize the tweet
    tweet_tokenized = token.tokenize(tweet2)

    # Create Porter Stemmer object
    stem = PorterStemmer()

    # Create stop words object
    st = stopwords.words('english')
    tweet_cleaned = []

    # Remove stopwords and punctuations and apply stemming
    for word in tweet_tokenized:
        if word not in st and word not in string.punctuation:
            stemmed_word = stem.stem(word)
            tweet_cleaned.append(stemmed_word)

    return tweet_cleaned


# A function to create a dictionary mapping {(word, label) --> frequency of that word in that label}
def build_freq(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    frequency = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            frequency[pair] = frequency.get(pair, 0) + 1
    return frequency


freqs = build_freq(train_x, train_y)


# A function to train our model
def train_naive_bayes(freqs, train_x, train_y):
    # Vocab is the set of unique words in our training set
    vocab = set([pair[0] for pair in freqs.keys()])

    # Get the number of unique words
    V = len(vocab)

    loglikelihood = {}

    # Calculate the logprior
    D = len(train_x)
    D_pos = np.sum(train_y)
    D_neg = D - D_pos
    logprior = np.log(D_pos) - np.log(D_neg)

    N_pos = N_neg = 0

    # Calculate N_pos and N_neg
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    # For each unique word
    for word in vocab:
        # Get its positive frequency
        pos_freq = freqs.get((word, 1), 0)
        # Get its negative frequency
        neg_freq = freqs.get((word, 0), 0)

        # Find probability that the word is positive
        p_w_pos = (pos_freq + 1) / (N_pos + V)
        # Find probability that the word is negative
        p_w_neg = (neg_freq + 1) / (N_neg + V)

        # Populate the loglikelihood dictionary
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


# A function to predict the sentiment of a tweet
def predict(tweet, logprior, loglikelihood):
    word_list = process_tweet(tweet)
    p = 0
    p += logprior

    # For each word in the tweet
    for word in word_list:
        # If the word has a loglikelihood
        if word in loglikelihood:
            # Get the total loglikelihood of all the words in the tweet
            p += loglikelihood[word]

    return p


# A function to test our model
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0
    y_hats = []

    # Fo reach tweet in the test set
    for tweet in test_x:
        if predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0

        y_hats.append(y_hat_i)

    # Error is the average of the absolute values of the differences between y_hats and test_y
    error = np.mean((np.abs(y_hats - test_y)))
    accuracy = 1 - error

    return accuracy


tweet = input("Enter your tweet: ")

if predict(tweet, logprior, loglikelihood) > 0:
    print("Positive sentiment")
else:
    print("Negative sentiment")

print("accuracy: ", test_naive_bayes(test_x, test_y, logprior, loglikelihood))
