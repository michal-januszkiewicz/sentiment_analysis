import nltk
import gzip
import re

from nltk.corpus import stopwords


###############################################################################
# Split a list in two lists.
###############################################################################
def split_list_in_two(a_list, test_partition):

  test_partition = (100 - test_partition) / 100
  boundary = int(len(a_list) * test_partition)
  return a_list[:boundary], a_list[boundary:]


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
    rating = convert_rating_to_sentiment(review['overall'])
    ratings.append(rating)

  return reviews, ratings


###############################################################################
# Convert star rating to sentiment number. 
# 1,2 stars -> 0 - negative
# 3 stars   -> 1 - neutral
# 4,5 stars -> 2 - positive
###############################################################################
def convert_rating_to_sentiment(rating):

  switcher = {
      1.0: 0,
      2.0: 0,
      3.0: 1,
      4.0: 2,
      5.0: 2,
  }

  return switcher.get(rating, "nothing")
