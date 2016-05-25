import nltk
import gzip
import re
import numpy as np

from nltk.corpus import stopwords


###############################################################################
# Extract only meaningful words from review text.
###############################################################################
def review_to_words(text):

  # Remove punctuation and numbers.
  text = re.sub("[^a-zA-Z]", " ", text) 

  # Split summary into array of lowercase words.
  words = [e.lower() for e in text.split()]

  # Convert list to set to speed up processing.
  stops = set(stopwords.words('english'))

  # Remove stop words.
  words = [w for w in words if not w in stops]

  # Convert array of words back into string.
  words = ' '.join(words)

  return words


###############################################################################
# Get only text reviews and star ratings from entire data set.
###############################################################################
def extract_reviews_and_rating(path, neutral_sentiment):

  g = gzip.open(path, 'r')
  reviews = []
  ratings = []

  for l in g:
    review = eval(l)
    rating = convert_rating_to_sentiment(review['overall'], neutral_sentiment)

    # Check if rating has an assigned sentiment.
    if rating is not None:

      # Get only meaningful words from review text.
      words = review_to_words(review['reviewText'])

      ratings.append(rating)
      reviews.append(words)

  # Convert python lists to numpy arrays.
  ratings = np.array(ratings)
  reviews = np.array(reviews)

  return reviews, ratings


###############################################################################
# Convert star rating to sentiment number. 
# 1,2 stars -> 0 - negative
# 3 stars   -> cut this off
# 4,5 stars -> 1 - positive
#           or
# 1,2 stars -> 0 - negative
# 3 stars   -> 1 - neutral
# 4,5 stars -> 2 - positive
###############################################################################
def convert_rating_to_sentiment(rating, neutral_sentiment):

  switcher = {
      1.0: 0,
      2.0: 0,
      3.0: None,
      4.0: 1,
      5.0: 1,
  }
  if neutral_sentiment:
    switcher = {
        1.0: 0,
        2.0: 0,
        3.0: 1,
        4.0: 2,
        5.0: 2,
    }

  return switcher.get(rating, None)
