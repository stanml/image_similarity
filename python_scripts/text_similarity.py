import itertools as it
import editdistance

# The segmentation algorithm can potentially extract text in an arbitrary order,
# therefore, the Levenshtein measure can classifiy text with exactly the same
# sentences as being different because of the order. This class gives a score
# based on matched segments by taking the Levenshtien distance between text
# segments to determine a match, and then returning the overall similarity score
# of matched segments within the image.

class TextSimilarity:
  def __init__(self, text_array1, text_array2, threshold=None):
    self.clean_text1 = self.__remove_blanks(text_array1)
    self.clean_text2 = self.__remove_blanks(text_array2)
    self.max_characters = self.__max_text(self.clean_text1, self.clean_text2)
    self.matches = []
    self.total_characters = []
    if threshold is None:
      self.match_threshold = 70
    else:
      self.match_threshold = threshold

  def return_matches(self):
    if (len(self.clean_text1) or len(self.clean_text2)) == 0:
      return False
    else:
      self.__get_matches()
      matches = sum(self.matches)
      return (float(matches) / self.max_characters) * 100

  def __get_matches(self):
    return map(lambda x: self.__matched_element(x), self.__combinations())

  def __matched_element(self, pair):
    score, matched_words = self.__n_text_matches(pair[0], pair[1])
    if score > self.match_threshold:
      self.matches.append(matched_words)

  def __n_text_matches(self, text1, text2):
    max_letters = self.__max_elems(text1, text2)
    matched_words = max_letters - editdistance.eval(text1, text2)
    score = (float(matched_words) / max_letters) * 100
    return score, matched_words

  def __max_elems(self, array1, array2):
    return max(len(array1), len(array2))

  def __max_text(self, text1, text2):
    return max(self.__count_characters(text1), self.__count_characters(text2))

  def __count_characters(self, array):
    return len(''.join(array))

  def __remove_blanks(self, text_array):
    return set(filter(None, text_array))

  def __combinations(self):
    return list(it.product(self.clean_text1, self.clean_text2))

