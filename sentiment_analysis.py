import nltk
import sys
import numpy as np

from helper_functions import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation


###############################################################################
# Main calculations function.
###############################################################################
def calculate(n_estimators = None, max_depth = None, max_features = None, 
    vectorizer_max_features = None, neutral_sentiment = None):

  # Read console parameters if function is not called from python.
  if (n_estimators is None):
    n_estimators            = int(sys.argv[1])
    max_depth               = int(sys.argv[2])
    max_features            = int(sys.argv[3])
    vectorizer_max_features = int(sys.argv[4])
    neutral_sentiment       = bool(sys.argv[5])

  # Set path to a dataset.
  path = "output.gz"

  # Get only text reviews and star ratings from entire data set.
  print('Extracting data...')
  reviews, ratings = extract_reviews_and_rating(path, neutral_sentiment)

  # Limit vocabulary size to 5000.
  vectorizer = CountVectorizer(
      analyzer = "word",   
      tokenizer = None,    
      preprocessor = None, 
      stop_words = None,   
      max_features = vectorizer_max_features
  )

  # Initialize a random forest classifier.
  forest = RandomForestClassifier(
      n_estimators = n_estimators, 
      max_depth = max_depth,
      max_features = max_features
  )

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
  accuracy = sum_accuracy / 10
  print('Average accuracy: ' + str(accuracy) + '\n')
  return accuracy

