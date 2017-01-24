from skimage.feature import (match_descriptors,plot_matches)
import itertools as it
import pdb

def matched_features(data, pair, names):
  descriptors1 = data[names[pair[0]]]['descriptors']
  descriptors2 = data[names[pair[1]]]['descriptors']
  return match_descriptors(descriptors1, descriptors2, cross_check=True)

def matched_text(data, pair, names):
  text1 = data[names[pair[0]]]['text']
  text2 = data[names[pair[1]]]['text']
  matches = [x for x in text1 if x in text2]
  return [x for x in matches if x is not None]

def compare(data, sets, names):
  results = {}
  for i in sets:
    results[i] = {
      'descriptors': matched_features(data, i, names),
      'text' : matched_text(data, i, names)
      }
  return results

def sets(data):
  return list(it.combinations([x * 1 for x in range(0, len(data))], 2))

def keypoint_matches(data, names, results, pair):
  n_key_points = len(data[names[pair[0]]]['descriptors'])
  n_kp_matches = len(results[pair]['descriptors'])
  return (float(n_kp_matches) / n_key_points) * 100

def text_matches(data, names, results, pair):
  n_words = (len(data[names[pair[0]]]['text']) + len(data[names[pair[1]]]['text'])) / 2
  n_text_matches = len(results[pair]['text'])
  return (float(n_text_matches) / n_words) * 100

def similarity_score(text_score, descriptor_score):
  return (text_score + descriptor_score) / 2

#def analysis(data, names, results, pair):
#  perc_text = text_matches(data, names, results, pair)
#  perc_desc = keypoint_matches(data, names, results, pair)
#  return "%s & %s: %d%% similarity" % (names[pair[0]], names[pair[1]], similarity_score(perc_text, perc_desc))
#
#def summary(data, names, results, combinations):
#  return map(lambda x: analysis(data, names, results, x), combinations)

def flagged_images(data, names, results, threshold, pair):
  perc_text = text_matches(data, names, results, pair)
  perc_desc = keypoint_matches(data, names, results, pair)
  sim_score = similarity_score(perc_text, perc_desc)
  if sim_score > threshold:
    return [names[pair[0]], names[pair[1]], str(round(sim_score, 1))]

def search_similarity(data, names, results, combinations, threshold):
  similar_images = map(lambda x: flagged_images(data, names, results, threshold, x), combinations)
  return [x for x in similar_images if x is not None]

def main(names, data, threshold):
  combinations = sets(data)
  results = compare(data, combinations, names)
  return search_similarity(data, names, results, combinations, threshold)


if __name__ == '__main__': main()
