from skimage import data
from skimage import io
from skimage.transform import downscale_local_mean
from skimage.feature import (match_descriptors,plot_matches, ORB)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
import itertools as it
import time
import numpy as np

def parameters(args):
  return sys.argv[1].split(','), map(int, sys.argv[2].split(','))

def image_array(filename_array):
  return map(lambda x: image_load(x), filename_array)

def image_load(filename):
  img = io.imread(filename)
  return rgb2gray(img)

def image_sizes(image_array):
 return map(lambda x: x.shape, image_array)

def target_size(image_sizes, downsize_factor):
  images_sum = map(lambda x: sum(x), image_sizes)
  max_idx = images_sum.index(max(images_sum))
  return tuple(x / downsize_factor for x in image_sizes[max_idx])

def downscale(image, downsize_factor):
  return downscale_local_mean(image, (downsize_factor,downsize_factor))

def calculate_downsize(image, target_size):
  size = np.ceil(np.asarray(image.shape, dtype=float) / np.asarray(target_size, dtype=float))
  return int(np.sum(size)) / 2

def orb_features(image, distance, extractor):
  extractor.detect_and_extract(image)
  keypoints = extractor.keypoints
  descriptors = extractor.descriptors
  return keypoints, descriptors

def data_hash(filename_array, image_array, target_size, downsize_factor, distance):
  images = {}
  for i in range(0, len(filename_array)):
    index = i + 1
    name = filename_array[i]
    image = image_array[i]
    gray = downscale(image, calculate_downsize(image, target_size))
    extractor = ORB(n_keypoints=300)
    keypoints, descriptors = orb_features(gray, distance, extractor)
    images[index] = {
      'name': name,
      'image': gray,
      'keypoints': keypoints,
      'descriptors': descriptors
    }
  return images

def matched_features(data, pair):
  descriptors1 = data[pair[0]]['descriptors']
  descriptors2 = data[pair[1]]['descriptors']
  return match_descriptors(descriptors1, descriptors2, cross_check=True)

def compare(data, sets):
  results = {}
  for i in sets:
    results[i] = matched_features(data, i)
  return results

def sets(image_features):
  return list(it.combinations([x * 1 for x in range(1, len(image_features) + 1)], 2))

def analysis(data, results, pair):
  n_key_points = max(len(data[pair[0]]['keypoints']), len(data[pair[1]]['keypoints']))
  n_matches = len(results[pair])
  percentage_matches = (float(n_matches)/n_key_points) * 100
  return "%s & %s: %d%% keypoints matched" % (data[pair[0]]['name'], data[pair[1]]['name'], round(percentage_matches,2))

def summary(data, results, combinations):
  return map(lambda x: analysis(data, results, x), combinations)

def plot_results(pairs, data, results):
  fig, ax = plt.subplots(nrows=len(pairs), ncols=1)
  plt.gray()
  for i in range(0, len(pairs)):
    plot_matches(ax[i], data[pairs[i][0]]['image'], data[pairs[i][1]]['image'], data[pairs[i][0]]['keypoints'], data[pairs[i][1]]['keypoints'], results[pairs[i]])
    ax[i].axis('off')
    ax[i].set_title("%s compared with %s" % (data[pairs[i][0]]['name'], data[pairs[i][1]]['name']))
  plt.show()

def print_results(matches):
  for i in matches:
    print i

def main():
  start_time = time.time()
  filename_array, params = parameters(sys.argv)
  images = image_array(filename_array)
  downsize_factor = params[0]
  distance = params[1]
  target = target_size(image_sizes(images), downsize_factor)
  data = data_hash(filename_array, images, target, downsize_factor, distance)
  combinations = sets(filename_array)
  results = compare(data, combinations)
  matches = summary(data, results, combinations)

  print_results(matches)
  print "--- Comparison took %s seconds ---" % (time.time() - start_time)

  plot_results(combinations, data, results)

if __name__ == '__main__': main()
