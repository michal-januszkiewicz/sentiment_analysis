import nltk

from helper_functions import *


###############################################################################
# Main function.
###############################################################################
def main():

  path = "output.gz"

  # Get only text reviews and star ratings from entire data set.
  reviews, ratings = extract_reviews_and_rating(path)


main()
