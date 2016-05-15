import nltk
import gzip
import re

from nltk.corpus import stopwords


###############################################################################
# Split a list in two.
###############################################################################
def split_list_in_half(a_list):
  half = int(len(a_list)/2)
  return a_list[:half], a_list[half:]


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
def extract_reviews_and_rating(path):

  g = gzip.open(path, 'r')
  reviews = []
  ratings = []

  for l in g:
    review = eval(l)

    # Get only meaningful words from review text.
    words = review_to_words(review['reviewText'])

    reviews.append(words)
    ratings.append(review['overall'])

  return reviews, ratings
