# Sentiment analysis

This is a sentiment analysis application to predict sentiment of amazon reviews using random forest algorithm.
The program uses amazon review data sets which can be downloaded from 
```
http://jmcauley.ucsd.edu/data/amazon/
```
> Image-based recommendations on styles and substitutes 
> J. McAuley, C. Targett, J. Shi, A. van den Hengel 
> SIGIR, 2015

Possibly other data sets could be used as well provided they have the same structure, are compressed with gzip and contain fields:  `reviewText` and `overall`.


## Installation

1. Install nltk 

  ```
  sudo apt-get install python3-nltk
  ```
2. Install sklearn and other modules

  ```
  sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base python3-pip
  sudo pip3 install scikit-learn
  ```

3. Install stopwords from nltk

  Put this line in the program(in main.py or sentiment_analysis.py) to run the installer:
  ```
  nltk.download()
  ```
  And download stopwords.  
  **Remove the line after installation.** 


## Data preparation

The data sets downloaded from the given link are very large. To shrink them down a little an additional program was written. Example of usage can be found in `extract_data.sh` file. You can also modify it for your own use.
The program needs following parameters:
  1. A data set path
  2. Output file name (note that `.gz` will be appended to it automatically)
  3. Number of reviews you want to extract


## Usage

The application(`sentiment_analysis.py`) can be run either with bash or with python.

Right now the application can use only a dataset that is in the same directory and is named: `output.gz`.  
**Always make sure you pass all parameters.**

1. Run from bash  
  You can find example usage in `run.sh` file.
  Remember to include all of the parameters:
  1. Number of estimators (integer)
  2. Maximal depth (integer)
  3. Maximal features (integer)
  4. Vocabulary size (integer)
  5. Include or exclude neutral sentiment (boolean)
  
  You can use this method to run the program just once or write a bash script to run it multiple times for different parameters.

2. Run from python
  This will involve running the `main.py` file which will run `sentiment_analysis.py` file multiple times with different parameters and save the results in a csv file.
  Feel free to experiment with params arrays and loop orders inside the `main.py` file.

  To run it just use:
  ```
  python3 main.py
  ```
