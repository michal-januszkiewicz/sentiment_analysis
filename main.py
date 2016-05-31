from sentiment_analysis import *
import timeit
import csv

n_estimators_array            = [200]
max_depth_array               = [600]
max_features_array            = [600]
vectorizer_max_features_array = [5000]
neutral_sentiment_array       = [False]

# Open and initialize the csv file with table headers.
with open('results.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ',
              quotechar='|', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['n_estimators'] + ['max_depth'] + ['max_features'] 
      + ['vocabulary_size'] + ['neutral_sentiment'] + ['time[s]'] + ['accuracy'])

  # Loop through all paramters.
  i = 0
  for n_estimators in n_estimators_array:
    for max_depth in max_depth_array:
      for max_features in max_features_array:
        for vectorizer_max_features in vectorizer_max_features_array:
          for neutral_sentiment in neutral_sentiment_array:

            # Run the algorithm for specific parameters.
            start = timeit.default_timer()
            accuracy = calculate('output.gz', n_estimators, max_depth, max_features, 
                vectorizer_max_features, neutral_sentiment)
            stop = timeit.default_timer()
            time = stop - start

            # Write the parameters and results to the csv file.
            writer.writerow([n_estimators] + [max_depth] + [max_features] 
                + [vectorizer_max_features] + [int(neutral_sentiment)] + [time] + [accuracy])
            i += 1
            print("Iteration: " + str(i))



