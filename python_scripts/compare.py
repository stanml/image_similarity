from text_similarity import TextSimilarity
from ast import literal_eval
import itertools as it
import cv2

# This class handles the comparison of the extracted features, it accepts a data
# hash that contains the extracted text and image features for each of the
# uploaded concepts. Every concept must be compared with every other concept in
# the survey components. OpenCV's FLANN matcher is used to match the extracted
# descriptor co-ordinates for each image, they are thresholded so that only the
# most salient features are compared. Text and Image features have their own
# independent threshold for both warning and error. Text analysis is handled
# through the TextSimilarity class.

class Compare():

  def __init__(self, data):
    self.data = data
    self.names = data.keys()
    self.combinations = self.__sets()
    self.results = {}
    self.index_params = dict(algorithm = 6, trees = 5)
    self.search_params = dict(checks=50)

  def get_results(self):
    return self.__search_similarity()

  def __flann_match(self, data, pair):
    descriptors1 = data[pair[0]]['descriptors']
    descriptors2 = data[pair[1]]['descriptors']
    flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=2)
    matches_mask = self.__matches_mask(matches)
    return self.__keypoint_matches(descriptors1, descriptors2, matches_mask)

  def __matches_mask(self, matches):
    matchesMask = [0 for i in xrange(len(matches))]
    for i in range(0, len(matches)):
      match = matches[i]
      if len(match) < 2:
        continue
      elif match[0].distance < 0.85*match[1].distance:
        matchesMask[i]=1
    return matchesMask

  def __keypoint_matches(self, descriptors1, descriptors2, mask):
    return (float(sum(mask)) / max(len(descriptors1), len(descriptors2))) * 100

  def __matched_text(self, data, pair):
    text1 = data[pair[0]]['text']
    text2 = data[pair[1]]['text']
    text_matcher = TextSimilarity(text1, text2)
    return text_matcher.return_matches()

  def __compare(self):
    for i in self.combinations:
      self.results[i] = {
        'descriptors': self.__flann_match(self.data, i),
        'text' : self.__matched_text(self.data, i),
        'threshold': self.__set_threshold(self.data, i)
        }

  def __set_threshold(self, data, pair):
    if data[pair[0]]['threshold'] == 'default' or data[pair[0]]['threshold'] == 'default':
      return 'default'
    else:
      return 'segment'

  def __sets(self):
    return list(it.combinations(self.names, 2))

  def __similarity_score(self, text_score, descriptor_score):
    if text_score == False:
      return descriptor_score
    else:
      return (text_score + descriptor_score) / 2

  def __single_hash(self, pair, perc_text, perc_desc, classification, threshold):
    return {
      'matchingImages': [pair[0], pair[1]],
      'similarityScore': classification,
      'text': round(perc_text, 1),
      'image': round(perc_desc, 1),
      'threshold': threshold
    }

  def __evaluate_threshold(self, threshold, perc_desc, perc_text):
    if threshold == 'default':
      image_error_threshold = 80
      text_error_threshold = 700
      image_warning_threshold = 65
      text_warning_threshold = 55
    else:
      image_error_threshold = 90
      text_error_threshold = 80
      image_warning_threshold = 75
      text_warning_threshold = 65
    perc_above_warning_threshold = perc_desc >= image_warning_threshold or perc_text >= text_warning_threshold
    perc_above_error_threshold = perc_desc >= image_error_threshold or perc_text >= text_error_threshold
    if perc_above_error_threshold:
      return 'error'
    elif perc_above_warning_threshold:
      return 'warning'

  def __flagged_images(self, pair):
    perc_text = self.results[pair]['text']
    perc_desc = self.results[pair]['descriptors']
    threshold = self.results[pair]['threshold']
    error = self.__evaluate_threshold(threshold, perc_desc, perc_text)
    if error != None:
      return self.__single_hash(pair, perc_text, perc_desc, error, threshold)


  def __search_similarity(self):
    self.__compare()
    similar_images = map(lambda pair: self.__flagged_images(pair), self.combinations)
    return [x for x in similar_images if x is not None]
