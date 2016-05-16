import nltk

from helper_functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


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

  # Limit vocabulary size to 5000.
  vectorizer = CountVectorizer(
      analyzer = "word",   
      tokenizer = None,    
      preprocessor = None, 
      stop_words = None,   
      max_features = 5000
  )

  train_data_features = vectorizer.fit_transform(reviews)
  train_data_features = train_data_features.toarray()

  # Initialize a random forest classifier.
  forest = RandomForestClassifier(n_estimators = 100)

  # Train classifier.
  forest = forest.fit(train_data_features, ratings)

  test_data_features = vectorizer.transform(test_reviews)
  test_data_features = test_data_features.toarray()

  # Use trained classifier to predict sentiment of test data.
  result = forest.predict(test_data_features)

  print(result)


main()
