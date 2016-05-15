import nltk

from helper_functions import *


###############################################################################
# Main function.
###############################################################################
def main():

  path = "output.gz"

  # Get only text reviews and star ratings from entire data set.
  reviews, ratings = extract_reviews_and_rating(path)

  # Split reviews and ratings into training and test data.
  reviews, test_reviews = split_list_in_half(reviews)
  ratings, test_ratings = split_list_in_half(ratings)


main()
