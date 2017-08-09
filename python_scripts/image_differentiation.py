from feature_extract import *
from compare import *
import sys
import json
import time

def inputs():
  #arguments = sys.stdin.readline().split(' ')
  #return arguments[0].split(','), arguments[1].split(',')
  return sys.argv[1].split(','), sys.argv[2].split(',')

def results_structure(names, data={}):
  for name in names:
    data[name] = {
    'index': names.index(name),
     'matchingImages': []
    }
  return data

def add_data(v, score, names, text, image, threshold):
  entry = {
    'id': v,
    'index': names.index(v),
    'similarity_score': score,
    'text_score': text,
    'image_score': image,
    'threshold': threshold
  }
  return entry

def output_data(matching_pairs, names):
  data = results_structure(names)
  for pair in matching_pairs:
    key = pair['matchingImages'][0]
    value = pair['matchingImages'][1]
    score = pair['similarityScore']
    text_score = pair['text']
    image_score = pair['image']
    threshold = pair['threshold']
    data[key]['matchingImages'].append(add_data(value, score, names, text_score, image_score, threshold))
    data[value]['matchingImages'].append(add_data(key, score, names, text_score, image_score, threshold))
  return data

def main():
  filepaths, ids = inputs()
  extractor = FeatureExtract(filepaths, ids, 'eng')
  results = Compare(extractor.get_features())
  output = output_data(results.get_results(), ids)
  sys.stderr.write(json.dumps(output))

if __name__ == '__main__' : main()
