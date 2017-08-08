import cv2
import numpy as np
from pytesseract import *
from PIL import Image

from matplotlib import pyplot as plt
from pylab import *

# This class segments concepts into text and product images. Firstly a binary
# threshold is applied so that the areas with the most intensity are retained.
# We then applly dilation so that text blocks are merged to form an area/segment.
# Segments that are too large, small or have low intensity values are discarded,
# the segments that remain are analysed for text, and if they contain > 60
# characters, are removed from the image. The dilation parameter is looped over
# so that we can account for a wide range of concepts structures.

class Segment(object):
  def __init__(self, image, language):
    self.image = image
    self.gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    self.threshold_limit = self.__average_colour_intensity(image)
    self.language = language
    self.text_mask = []

  def __threshold(self, image):
    _,thresh = cv2.threshold(image,self.threshold_limit,255,cv2.THRESH_BINARY_INV)
    return thresh

  def __average_colour_intensity(self, image):
    av_colour_per_row = np.average(image, axis=0)
    return int(np.average(av_colour_per_row))

  def __is_text(self, segment):
    segment_hist = self.__normalise(segment.histogram()[:-1])
    abs_diff = sum(abs(segment_hist - self.text_mask))
    print 'image_sum: %f, abs_diff: %f' % (sum(self.text_mask),abs_diff)
    if abs_diff < sum(self.text_mask):
      return True
    else:
      return False

  def __normalise(self, array):
    x = np.array(array)
    return x / np.linalg.norm(x)

  def segment_text(self):
    image2 = Image.fromarray(self.image).convert('L')
    thresh = self.__threshold(self.gray)
    image1 = self.image
    all_text = []
    for i in range(16,5,-2):
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
      dilated = cv2.dilate(thresh,kernel,iterations = i)
      _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
      imshow(thresh)
      show()
      for contour in contours:
        [x,y,width,height] = cv2.boundingRect(contour)
        # discard areas that are too large
        if height>400 and width>400:
          continue
        # discard areas that are too small
        if height<50 or width<50:
          continue
        bbox = (x, y, x+width, y+height)
        segment = image2.crop(bbox)
        if len(self.text_mask) == 0:
          text = image_to_string(segment, lang=self.language, config='-psm 3')
          all_text.append(text)
          if len(text) > 60:
            self.text_mask = self.__normalise(segment.histogram()[:-1])
            image1[y:y+height, x:x+width, :] = self.threshold_limit
            thresh[y:y+height, x:x+width] = self.threshold_limit
            image2 = Image.fromarray(image1).convert('L')
        elif self.__is_text(segment):
        #if np.unique(np.array(segment)).size < 200 and segment.getextrema()[1] > 215:
          text = image_to_string(segment, lang=self.language, config='-psm 3')
          all_text.append(text)
          if len(text) > 60:
            image1[y:y+height, x:x+width, :] = self.threshold_limit
            thresh[y:y+height, x:x+width] = self.threshold_limit
            image2 = Image.fromarray(image1).convert('L')
    return image1, all_text
