#
# Call the modules' API from python
#
# (c) 2025 Jean-Olivier Irisson, GNU General Public License v3

# list images
from glob import glob
images = sorted(glob('images/*'))

import requests

## Classifier ----

def walton_url(model):
  """Return the URL of the given model on the Walton cluster"""
  return 'https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-'+model+'/exposed/main/v2/models/zooprocess_multiple_'+model+'/predict/'
walton_url('foo')

# call the classifier with the first image
r = requests.post(walton_url('classifier'),
      params={'bottom_crop': 31},
      files={'images': open(images[0], 'rb')})
# parse the JSON response into a Python dict
r_clas = r.json()
r_clas
type(r_clas)

# zip the imaged and send them
import zipfile
import os
with zipfile.ZipFile('images.zip', 'w') as zipf:
  for img in images:
    # add each image, and add it to the root of the zip
    zipf.write(img, os.path.basename(img))
  
# now send the zip instead of a single image
r = requests.post(walton_url('classifier'),
      params={'bottom_crop': 31},
      # specify file     name         content                    type
      files={'images': ('images.zip', open('images.zip', 'rb'), 'application/zip')})
r.json()

## Separator ----

# just as before, call the separator, with one additional argument
r = requests.post(walton_url('separator'),
      params={'min_mask_score':0.9, 'bottom_crop': 31},
      files={'images': open(images[0], 'rb')})
r_sep = r.json()
r_sep

# plot the result, as we did in R
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# read the images
img = mpimg.imread(images[0])
# convert it into an array
img = np.array(img)
# extract the coordinates of the separation line(s)
coords = tuple(r_sep['predictions'][0]['separation_coordinates'])
# set those to pure red
img[:,:,0][coords] = 255
img[:,:,1][coords] = 0
img[:,:,2][coords] = 0
# and ploth the result
plt.axis('off')
imgplot = plt.imshow(img, )
plt.show()

# zip a few images
with zipfile.ZipFile('images_few.zip', 'w') as zipf:
  for img in images[0:3]:
    zipf.write(img, os.path.basename(img))

# and send this zip
r = requests.post(walton_url('separator'),
      params={'bottom_crop': 31},
      files={'images': ('images_few.zip', open('images_few.zip', 'rb'), 'application/zip')})
r.json()


## Remeasure separated particles ----

from skimage import io
from skimage.measure import label, regionprops_table

# separate objects on one image
r = requests.post(walton_url('separator'),
      params={'min_mask_score':0.9, 'bottom_crop': 31},
      files={'images': open(images[0], 'rb')})
r_sep = r.json()

# read the image, in greyscale
img = io.imread(images[0], as_gray=True)
# plot it
plt.imshow(img, cmap='gray', vmin=0, vmax=1); plt.show()

# white out the pixels corresponding to the separation line
coords = tuple(r_sep['predictions'][0]['separation_coordinates'])
img[coords] = 1
plt.imshow(img, cmap='gray', vmin=0, vmax=1); plt.show()
  
# remove the scale bar
dims = img.shape
img = img[0:(dims[0]-31),:]
plt.imshow(img, cmap='gray', vmin=0, vmax=1); plt.show()

# binarize the image (the background is pure white)
img_bin = img < 1
plt.imshow(img_bin); plt.show()
# NB: some artefacts (small dots) are due to the JPEG compression

# define objects on the binary image (=connected components)
img_lab = label(img_bin)
plt.imshow(img_lab); plt.show()

# get a bunch of properties of each object
props = regionprops_table(label_image=img_lab, intensity_image=img,
  properties=[
    'area', 'area_filled', 'equivalent_diameter_area',
    'axis_major_length', 'axis_minor_length',
    'eccentricity',
    'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std',
    'moments_central', 'moments_hu',
    'moments_weighted_central', 'moments_weighted_hu'
  ])

# convert it into a table and clean the very small objects (compression noise)
import pandas as pd
props_df = pd.DataFrame(props)
props_df = props_df.query("area > 10")
props_df

# The same can be done for images of single objects, without the separation step.
# This provides new descriptors of the images, resembling those from ZooProcess

# Alternatively, one can re-create the sep.gif file that ZooProcess uses,
# by drawing the white lines on a black background in the original position of
# each image. Then ZooProcess cand be re-run and will extract properties 
# completely compatible with the other images.

