#!/bin/bash
#
# Call the modules' API from shell scripts, through curl
#
# (c) 2025 Jean-Olivier Irisson, GNU General Public License v3

## Classifier ----

# 1. copy the curl command from the web interface
# 2. change the path to point to the file on your local drive
# 3. execute the command in a shell
curl -X 'POST' \
  'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-classifier/exposed/v2/models/zooprocess_multiple_classifier/predict/?bottom_crop=31' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'images=@images/s_8077.jpg;type=image/jpeg'

# we can actually drop some headers and type, and it still works
curl -X 'POST' \
  'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-classifier/exposed/v2/models/zooprocess_multiple_classifier/predict/?bottom_crop=31' \
  -F 'images=@images/s_8077.jpg'

# now put that in a loop
for img in $(ls images/)
do
  curl -X 'POST' \
    'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-classifier/exposed/v2/models/zooprocess_multiple_classifier/predict/?bottom_crop=31' \
    -F "images=@images/$img"
  # NB: you need double quotes for the variable $img to be interpreted
done

# or zip the files
zip -jX0 images.zip images/*.jpg
# NB:
# -j : allows to put all images at the base of the zip file,
#      with no folder structure, which is what is expected by the API.
# -X : do not include extended file attributes
# -0 : do not compress the images (they are already compressed by JPEG,
#      there is very little to be gained and it will just slow things down)

# copy the command when uploading a zip
curl -X 'POST' \
  'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-classifier/exposed/v2/models/zooprocess_multiple_classifier/predict/?bottom_crop=31' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'images=@images.zip;type=application/zip' # note the change of type


## Separator ----

# exactly the same
curl -X 'POST' \
  'https://inference-walton.cloud.imagine-ai.eu/system/services/zooscan-multiple-separator/exposed/v2/models/zooprocess_multiple_separator/predict/?min_mask_score=0.9&bottom_crop=31' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'images=@images/s_8077.jpg;type=image/jpeg'
# NB: the execution is longer since the model is (much) more complicated
