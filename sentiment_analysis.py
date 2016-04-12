import json
import gzip

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    eval(l)

path = "/home/michal/amazon_data_sets/reviews_Apps_for_Android.json.gz"
parse(path)
