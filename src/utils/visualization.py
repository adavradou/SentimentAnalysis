
import matplotlib.pyplot as plt
import os

import seaborn as sns
import nltk
from nltk.corpus import stopwords


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
  metric  = model_hist.history['accuracy']

  plt.clf()
  plt.plot(metric)
  plt.xlabel('Epochs')
  plt.ylabel('accuracy')
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


