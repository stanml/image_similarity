import feature_extract
import compare
import sys
import time
import matplotlib.pyplot as plt
import pdb
import numpy as np

def parameters():
  return sys.argv[1].split(','), map(int, sys.argv[2].split(','))

def get_data(filenames, downsize_factor, distance):
  data = {}
  start_time = time.time()
  for i in range(0, len(filenames)):
    filename = filenames[i]
    features = feature_extract.main(filename, downsize_factor, distance)
    data[filename] = {'descriptors' : features[0], 'text' : features[1], 'image': features[2]}
  print "--- Extraction took %s seconds ---" % (time.time() - start_time)
  return data

def results_to_string(results):
  flat_results = [item for result in results for item in result]
  return (' ').join(flat_results)

def plot_results(results, data):
  n = len(results)
  if n > 0:
    fig, axarr = plt.subplots(n)
    if n > 1:
      for i in range(0, n):
        image = np.concatenate((data[results[i][0]]['image'], data[results[i][1]]['image']), axis=1)
        axarr[i].imshow(image, cmap='gray')
        axarr[i].set_title("%s and %s are too similar (%s%%)" % (results[i][0], results[i][1], results[i][2]))
        axarr[i].axes.get_xaxis().set_visible(False)
        axarr[i].axes.get_yaxis().set_visible(False)
    else:
      image = np.concatenate((data[results[0]]['image'], data[results[1]]['image']), axis=1)
      axarr.imshow(image, cmap='gray')
      axarr.set_title("%s and %s are too similar" % (results[0], results[1]))
      axarr.axes.get_xaxis().set_visible(False)
      axarr.axes.get_yaxis().set_visible(False)
    plt.gray()
    plt.show()
  else:
    return

def main():
  filenames, params = parameters()
  data = get_data(filenames, params[0], params[1])
  results = compare.main(filenames, data, params[2])
  sys.stdout.write(results_to_string(results))
  plot_results(results, data)

if __name__ == '__main__' : main()
