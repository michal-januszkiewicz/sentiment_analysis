from sentiment_analysis import *
import csv

n_estimators_array            = [100, 1000]
max_depth_array               = [100, 1000]
max_features_array            = [100, 1000]
vectorizer_max_features_array = [5000, 8000]
neutral_sentiment_array       = [True, False]

# Open and initialize the csv file with table headers.
with open('results.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ',
              quotechar='|', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['n_estimators'] + ['max_depth'] + ['max_features'] 
      + ['vocabulary_size'] + ['neutral_sentiment'] + ['accuracy'])

  # Loop through all paramters.
  for n_estimators in n_estimators_array:
    for max_depth in max_depth_array:
      for max_features in max_features_array:
        print(max_features)
        for vectorizer_max_features in vectorizer_max_features_array:
          for neutral_sentiment in neutral_sentiment_array:

            # Run the algorithm for specific parameters.
            accuracy = calculate(n_estimators, max_depth, max_features, 
                vectorizer_max_features, neutral_sentiment)

            # Write the parameters and results to the csv file.
            writer.writerow([n_estimators] + [max_depth] + [max_features] 
                + [vectorizer_max_features] + [int(neutral_sentiment)] + [accuracy])



