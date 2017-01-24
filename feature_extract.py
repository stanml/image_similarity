from skimage.transform import downscale_local_mean
from skimage.feature import ORB
from PIL import Image
from pytesseract import *
import numpy as np

def load_image(filename):
  return Image.open(filename)

def pre_process(image, downsize_factor):
  gray = np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0])
  return downscale_local_mean(gray, (downsize_factor,downsize_factor))

def orb_features(image, distance, extractor):
  extractor.detect_and_extract(image)
  keypoints = extractor.keypoints
  descriptors = extractor.descriptors
  return descriptors

def extract_text(image):
  text = image_to_string(image).split()
  clean_text = map(lambda x: clean_text_array(x), text)
  return [x for x in clean_text if x is not None]

def clean_text_array(string):
  clean_data =  "".join(e for e in string if e.isalnum())
  if len(clean_data) > 1:
    return clean_data

def extract_features(filename, downsize_factor, distance):
  image = load_image(filename)
  gray = pre_process(image.convert('L'), downsize_factor)
  extractor = ORB(n_keypoints=500)
  descriptors = orb_features(gray, distance, extractor)
  text = extract_text(image)
  return descriptors, text, gray

def main(filename, downsize_factor, distance):
  data = extract_features(filename, downsize_factor, distance)
  return data

if __name__ == '__main__' : main()
