import nltk
import numpy as np

from helper_functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


###############################################################################
# Main function.
###############################################################################
def main():

  path = "output.gz"

  # Get only text reviews and star ratings from entire data set.
  print('Extracting data...')
  reviews, ratings = extract_reviews_and_rating(path)

  # Limit vocabulary size to 5000.
  vectorizer = CountVectorizer(
      analyzer = "word",   
      tokenizer = None,    
      preprocessor = None, 
      stop_words = None,   
      max_features = 5000
  )

  # Initialize a random forest classifier.
  forest = RandomForestClassifier(n_estimators = 100, max_depth = 1, max_features = 100)

  # Create bag of words features.
  train_data_features = vectorizer.fit_transform(reviews)
  train_data_features = train_data_features.toarray()

  # Create bag of words features.
  test_data_features = vectorizer.transform(reviews)
  test_data_features = test_data_features.toarray()

  # Prepare train and test indices for tenfold cross validation.
  kf = cross_validation.KFold(len(reviews), n_folds = 10)
  sum_accuracy = 0

  # Tenfold cross validation loop.
  for train, test in kf:

    # Convert python lists to numpy arrays.
    train = np.array(train)
    test = np.array(test)

    # Train classifier.
    print('Training classifier...')
    forest = forest.fit(train_data_features[train], ratings[train])

    # Use trained classifier to predict sentiment of test data.
    print('Processing test data...')
    result = forest.predict(test_data_features[test])

    # Calculate prediction accuracy.
    accuracy = accuracy_score(ratings[test], result)
    print('Accuracy: ' + str(accuracy) + '\n')
    
    # Sum each score to calculate average.
    sum_accuracy += accuracy


  # Display average accuracy for tenfold cross validation.
  print('Average accuracy: ' + str(sum_accuracy / 10))

main()
