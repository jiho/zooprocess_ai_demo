#
# Currently, the AI modules allow to process cropped images ("vignettes"
# in ZooProcess parlance). It allows to compute separation lines on those and
# they can be remeasured, with scikit-image for example. But the measurements
# are not fully compatible with ZooProcess.
#
# A solution is to draw those separation lines on a full scan, hence recreating 
# the file scanID_tot_1_sep.gif that ZooProcess can use to recompute the
# "vignettes" with the separationtaken into account.
#
# To do so we:
# 1. read the full image and mask created by ZooProcess
# 2. crop the objects to exactly their bouding box
# 3. send these tight crops to the AI modules
# 4. extract the separation lines and draw them within the bounding box of each
#    object detected as a multiple but on a full original image
#
# (c) 2025 Jean-Olivier Irisson, GNU General Public License v3

import requests
import pandas as pd
import zipfile
import os
from glob import glob
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

## 1. Read a full scan ----

# path to a scan inside a ZooProcess project
scan = '/Users/jiho/Desktop/jb19900214_tot_1/'
# NB: must end with '/'

# get the scan ID from the path
scan_id = scan.split("/")[-2]

# read the measurements file which contains the name of each vignette and
# its coordinates in the original scan
meas_path = scan + scan_id + '_meas.txt'
m = pd.read_csv(meas_path, sep='\t',
      usecols=['BX', 'BY', 'Width', 'Height'])

# read the full greyscale image
vis_path = scan + scan_id + '_vis1.tif'

# it may be zipped; unzip it if necessary
if not os.path.exists(vis_path):
  zip_path = scan + scan_id + '_vis1.zip'
  with zipfile.ZipFile(zip_path) as zipf:
    zipf.extractall(scan)

Image.MAX_IMAGE_PIXELS = 10**9 # allow to read large images
vis = Image.open(vis_path)
# extract the pixel array
vis = np.array(vis)

# read the pixel mask image
msk_path = scan + scan_id + '_msk1.gif'
msk = Image.open(msk_path)
msk = np.array(msk)


## 2. Crop each object ----

# create a temporary directory to store them
import tempfile
tmp_dir = tempfile.TemporaryDirectory()
nb_objects = m.shape[0]

# for each object in the measurements file
for i in range(nb_objects):
  crop_vis = vis[m['BY'][i]:m['BY'][i]+m['Height'][i],m['BX'][i]:m['BX'][i]+m['Width'][i]].copy()
  crop_msk = msk[m['BY'][i]:m['BY'][i]+m['Height'][i],m['BX'][i]:m['BX'][i]+m['Width'][i]].copy()
  # NB: copy to new arrays, otherwise we alter the full array and may mask objects
  
  # white out the background around the object
  crop_vis[crop_msk == 255] = 255
  # plt.imshow(crop_vis); plt.show()
  
  # convert and save into an RGB image (required by the module)
  crop_vis = np.stack([crop_vis,crop_vis,crop_vis], axis=2)
  img = Image.fromarray(crop_vis, mode='RGB')
  img.save(os.path.join(tmp_dir.name, str(i) + '.png'))


## 3a. Classify the crops ----

# zip all images to process them faster
with zipfile.ZipFile('scan_full.zip', 'w') as zipf:
  for i in range(nb_objects):
    zipf.write(os.path.join(tmp_dir.name, str(i) + '.png'), str(i)+'.png')

# send the zip to be classified through the API
base_url = 'http://marie.obs-vlfr.fr'
r = requests.post(base_url+':5000'+'/v2/models/zooprocess_multiple_classifier/predict/',
      params={'bottom_crop': 0},
      files={'images': ('scan_full.zip', open('scan_full.zip', 'rb'), 'application/zip')})
# and get the classification results
classif = r.json()

# define which images are considered as containing multiple objects, based on
# the score (0.4 allows to increase recall of multiples compared to 0.5)
is_multiple = [score > 0.4 for score in classif['scores']]


## 3b. Separate objects within crops considered as multiple ----

# create a zip with only the images considered to be contain multiple objects
idx_multiple = [i for i in range(nb_objects) if is_multiple[i]]
with zipfile.ZipFile('scan_multiples.zip', 'w') as zipf:
  for i in idx_multiple:
    zipf.write(os.path.join(tmp_dir.name, str(i) + '.png'), str(i)+'.png')

# send those to the separator API
r = requests.post(base_url+':5001'+'/v2/models/zooprocess_multiple_separator/predict/',
      params={'min_mask_score':0.9, 'bottom_crop': 0},
      files={'images': ('scan_multiples.zip', open('scan_multiples.zip', 'rb'), 'application/zip')})
pred = r.json()['predictions']

# we can remove the temporary crops
tmp_dir.cleanup()


## 4. Create a separation mask for the full image ----

# create an empty "sep" image
sep = np.zeros_like(msk)

# successively draw the separation lines, for each processed vignette
for p in pred:
  # if seom separations were made
  if (len(p['separation_coordinates'][0]) > 0):
    # get the index this crop
    # NB: images are named as integer indexes
    i = int(p['name'][0:-4])

    # recompute the coordinates, shifted into the dimensions of the full image
    x = p['separation_coordinates'][1] + m['BX'][i]
    y = p['separation_coordinates'][0] + m['BY'][i]

    # set these to be masked
    sep[tuple([y, x])] = 255
    
    # # check: add the separation line on the vis image and plot the resulting crop
    # vis[tuple([y, x])] = 0
    # crop_vis = vis[m['BY'][i]:m['BY'][i]+m['Height'][i],m['BX'][i]:m['BX'][i]+m['Width'][i]].copy()
    # plt.imshow(crop_vis); plt.show()

# save the image
sep_img = Image.fromarray(sep)
sep_img.save(scan_id + '_sep.gif')
