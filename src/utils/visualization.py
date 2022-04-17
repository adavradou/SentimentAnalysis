
import matplotlib.pyplot as plt
import os

import seaborn as sns
import nltk
from nltk.corpus import stopwords

from collections import Counter



"""
Plots the model's loss during training.
Input: model history, path to save the plots.
"""
def plot_loss_graph(model_hist):
  loss = model_hist.history['loss']
  val_loss = model_hist.history['val_loss']

  # Plot loss graph
  plt.clf()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.xlabel('Epochs')
  plt.ylabel('Binary crossentropy')
  # plt.ylim(0, model_hist.history['loss'][1]+0.01)
  plt.title('Model loss')
  plt.legend(['Train', 'Validation'], loc='center right')
  plt.show()


"""
Plots the training history, with respect to a specific metric e.g. PSNR.
Input: model history, path to save the plot, the name of the metric.
"""  

def plot_metric_graph(model_hist):
  accuracy  = model_hist.history['accuracy']
  val_accuracy  = model_hist.history['val_accuracy']

  plt.clf()
  plt.plot(accuracy)
  plt.plot(val_accuracy)
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
  plt.title('Model accuracy')
  plt.legend(['Train', 'Validation'], loc='center right')
  plt.show()



"""
Return the length of a tweet.
Input: tweet(str)
Output: length(int)
"""
def tweet_length(tweet):
  return len([token for token in tweet.split()])



"""
Removes the emojis from a tweet.
Input:tweet for cleaning from emojis(str)
Output: tweet without emojis(str)
"""
def visualize_tweet_length(data):
    tweet_lengths = [tweet_length(tweet) for tweet in data.tweet.tolist()]
    sns.distplot(tweet_lengths)

    # Unique words
    unique_words = set([token for tweet in data.tweet for token in tweet.split()])
    print("Total Unique Words:", len(unique_words))

    # Counting Total Words and Stop Words
    nltk.download("stopwords")
    stop_words = stopwords.words("english")
    total_words = [token for tweet in data.tweet for token in tweet.split()]
    total_stop_words = [token for tweet in data.tweet for token in tweet.split() if token in stop_words]
    print('Total words', len(total_words))
    print('Total stop words', len(total_stop_words))
    print('Ratio of total words to total stop words:', len(total_words)/len(total_stop_words))


"""
Maps the labels to text.
0 -> NEGATIVE
2 -> NEUTRAL
4 -> POSITIVE
"""
def decode_sentiment(label):
    decode_map = {0: "NEGATIVE", 1: "POSITIVE"}
    return decode_map[int(label)]


"""
Visualizes the distribution of the labels in a bar plot.
"""
def visualize_label_distribution(data):

  decoded_data = data['sentiment'].apply(lambda x: decode_sentiment(x))

  target_cnt = Counter(decoded_data)

  plt.figure(figsize=(16,8))
  plt.bar(target_cnt.keys(), target_cnt.values())
  plt.title("Dataset labels distribution")
