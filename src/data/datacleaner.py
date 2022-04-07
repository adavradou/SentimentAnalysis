import re
import wordninja, contractions, emoji

"""
Removes the emojis from a tweet.
Input:tweet for cleaning from emojis(str)
Output: tweet without emojis(str)
"""
def strip_emoji(tweet):
  new_tweet = re.sub(emoji.get_emoji_regexp(), r"", tweet)
  return new_tweet.strip()


"""
Removes the urls from a tweet.
Input:tweet for cleaning from urls(str)
Output: tweet without urls(str)
"""
def strip_urls(tweet):
  new_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, 
                     flags=re.MULTILINE)
  return new_tweet.strip()


"""
Removes the tags from a tweet.
Input:tweet for cleaning from tags(str)
Output: tweet without tags(str)
"""
def remove_tags(tweet):
  return " ".join([token for token in tweet.split() if not token.startswith("@")])


"""
Performs tweet preprocessing, by removing emojis, urls and tags.
Input:tweet for preprocessing(str)
Output: preprocessed tweet(str)
"""
def preprocess_tweet(tweet):
  tweet = remove_tags(strip_emoji(strip_urls(tweet)))
  tweet = contractions.fix(" ".join(wordninja.split(tweet)))
  tweet = [token.lower() for token in tweet.split() if (len(set(token))>1)]
  return " ".join(tweet)