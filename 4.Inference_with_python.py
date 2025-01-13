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
  return 'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-'+model+'/exposed/v2/models/zooprocess_multiple_'+model+'/predict/'
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

# TODO measure with scikit image

# zip a few images
with zipfile.ZipFile('images_few.zip', 'w') as zipf:
  for img in images[0:3]:
    zipf.write(img, os.path.basename(img))

# and send this zip
r = requests.post(walton_url('separator'),
      params={'bottom_crop': 31},
      files={'images': ('images_few.zip', open('images_few.zip', 'rb'), 'application/zip')})
r.json()


