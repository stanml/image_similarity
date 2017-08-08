from PIL import Image
from pytesseract import *
import numpy as np
import requests
import cv2
from segment import Segment
from urllib2 import urlopen

# This class extracts features from each concept by using the image url passed
# in. Firstly we attempt to segment each image, if this is not possible we
# extract text and features from the whole concept. The extraction methiod is
# handles by OpenCV and is equivalent to the scale inavriant feature transform
# (SIFT) algorithm which is the state of the art methood for extracting image
# features. Images are resized so that the largest dimension is 1200px, this is
# so we can perform the same process on all inputs.

class FeatureExtract:

  def __init__(self, filepaths, ids, language):
    self.filepaths = filepaths
    self.ids = ids
    self.downsize_factor = 1
    self.distance = 1
    self.language = language
    self.data = {}
    self.threshold_type = 'segment'

  def get_features(self):
    return self.__extract_features(self.data)

  def __extract_features(self, data):
    for i in range(0, len(self.filepaths)):
      if self.filepaths[i] != 'False':
        image = self.__load_image(self.filepaths[i])
        seg = Segment(image, self.language)
        image_no_text, text = seg.segment_text()
        combined_text = ' '.join(text)
        if len(combined_text) == 0 or combined_text.isspace():
            text_image = self.__pre_process(image)
            text = image_to_string(text_image, lang=self.language, config='-psm 3')
            self.threshold_type = 'default'
        keypoints, descriptors = self.__ORB_features(image_no_text)
        if descriptors is None:
            descriptors = np.array([])
        data[self.ids[i]] = {
        'text': text,
        'descriptors': descriptors,
        'threshold': self.threshold_type
        }
    return data

  def __load_image(self, filename):
    #request = urlopen(filename)
    #img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    #image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.imread(filename)
    width, height, _ = image.shape
    if (height < 1200) and (height > width):
        return self.__resize_image(image, height, width, 1200)
    elif (width < 1200) and (width > height):
        return self.__resize_image(image, width, height, 1200)
    else:
        return image

  def __resize_image(self, image, dimension1, dimension2, new_size):
    ratio = float(new_size) / dimension1
    dim = (new_size, int(dimension2 * ratio))
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

  def __ORB_features(self, image):
    detector = cv2.AKAZE_create()
    detector.setThreshold(0.004)
    return detector.detectAndCompute(image,None)

  def __pre_process(self, image):
    return Image.fromarray(image).convert('L')
