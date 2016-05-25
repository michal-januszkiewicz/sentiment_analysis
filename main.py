from sentiment_analysis import *

n_estimators_array            = [100, 1000]
max_depth_array               = [100, 1000]
max_features_array            = [100, 1000]
vectorizer_max_features_array = [5000, 10000]
neutral_sentiment_array       = [True, False]

for n_estimators in n_estimators_array:
  for max_depth in max_depth_array:
    for max_features in max_features_array:
      for vectorizer_max_features in vectorizer_max_features_array:
        for neutral_sentiment in neutral_sentiment_array:
          calculate(n_estimators, max_depth, max_features, 
              vectorizer_max_features, neutral_sentiment)



