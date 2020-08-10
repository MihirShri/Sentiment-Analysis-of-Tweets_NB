"""
Author @ Mihir_Srivastava
Dated - 17-07-2020
File - Sentiment_analysis_of_tweets_NB
Aim - To do the sentiment analysis of tweets from the twitter_samples DataSet available in the nltk library and classify
them as positive or negative tweets using sklearn's Multinomial Naive Bayes.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import string
import warnings
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import naive_bayes


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


# A function to extract the necessary features from a tweet [positive frequency sum, negative frequency sum]
def extract_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    word_list = process_tweet(tweet)

    # 2 elements in the form of a 1 x 2 vector
    x = np.zeros((1, 2))

    # loop through each word in the list of words
    for word in word_list:
        # increment the word count for the positive label 1
        x[0, 0] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0
        x[0, 1] += freqs.get((word, 0.0), 0)

    return x


warnings.filterwarnings('ignore')

# collect the features 'x' and stack them into a matrix 'X'
X_train = np.zeros(((len(train_x)), 2))
for i in range(len(train_x)):
    X_train[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
y_train = np.array(train_y)

# Create Multinomial Naive Bayes object
model = naive_bayes.MultinomialNB(alpha=1)

# Train the model
model.fit(X_train, y_train)

# Extract features of the test set
X_test = np.zeros(((len(test_x)), 2))
for i in range(len(test_x)):
    X_test[i, :] = extract_features(test_x[i], freqs)

y_test = np.array(test_y)

tweet = input("Enter your tweet: ")
tweet_final = extract_features(tweet, freqs)

print(model.predict(tweet_final))

if model.predict(tweet_final) == 1.:
    print("Positive sentiment")
else:
    print("Negative sentiment")

accuracy = model.score(X_test, y_test)
print("accuracy: " + str(accuracy))
