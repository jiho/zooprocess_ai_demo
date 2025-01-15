#
# Call the modules' API from R
#
# (c) 2025 Jean-Olivier Irisson, GNU General Public License v3


library("httr2")
library("curl")
library("tidyverse")

# list example image files
images <- list.files("images", full.names=TRUE)


## Classifier ----

# Send a single image
r <- request("https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-classifier/exposed/main/v2/models/zooprocess_multiple_classifier/predict/") |>
  # add parameter
  req_url_query(bottom_crop=31) |>
  # set type of query (this is a POST since there is a body)
  req_method("POST") |>
  # upload the image
  req_body_multipart(images=form_file(images[1])) |>
  # perform the request
  req_perform()

# the API replies with a json object
# parse it as an R list
r_clas <- resp_body_json(r, simplifyVector=TRUE)
r_clas

# Send a zip file
# zip the images together
# NB: we are using the same options as in the shell script
#     the important one is -j to make sure all images are at the root of the zip
zip(zipfile="images.zip", files=images, flags="-jX0")

# make the same query but with the zip file
r <- request("https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-classifier/exposed/main/v2/models/zooprocess_multiple_classifier/predict/") |>
  req_url_query(bottom_crop=31) |>
  req_method("POST") |>
  # explicitly specify the type as application/zip to make sure it is understood correctly
  req_body_multipart(images=form_file("images.zip", type="application/zip")) |>
  req_perform()

resp_body_json(r, simplifyVector=TRUE)
# in this case we get vectors as response


## Separator ----

# Single image
r <- request("https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-separator/exposed/main/v2/models/zooprocess_multiple_separator/predict/") |>
  # there is now one more argument to specify
  req_url_query(min_mask_score=0.9, bottom_crop=31) |>
  req_method("POST") |>
  req_body_multipart(images=form_file(images[1])) |>
  req_perform()

# parse the result (more options to get the data in the correct shape)
r_sep <- resp_body_json(r, simplifyVector=TRUE, simplifyDataFrame=FALSE, simplifyMatrix=TRUE)
r_sep
# now the response includes two components and `predictions` is a list, since we can send several images at once
# get only the first element
r_sep <- r_sep$predictions[[1]]

# Display the result = the separation line on an image
library("jpeg")
library("grid")

# read image
img <- readJPEG(images[1])

# define the separation coordinates as a matrix of indexes of the image
# NB: indexes start at 0 in the response (it comes from a Python script) so we need to add 1 since R's indexes start at 1.
m <- t(r_sep$separation_coordinates)+1
# at these indexes, set the colour to be pure red
img[,,1][m] <- 1
img[,,2][m] <- 0
img[,,3][m] <- 0
# plot the modified image as a raster
grid.newpage()
grid.raster(img, interpolate=TRUE)


# Send a zip file
# zip only a few images for this test (otherwise the server may time out)
zip(zipfile="images_few.zip", files=images[1:3], flags="-jX0")

# make the same query but with the zip file
r <- request("https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-separator/exposed/main/v2/models/zooprocess_multiple_separator/predict/") |>
  req_url_query(min_mask_score=0.9, bottom_crop=31) |>
  req_method("POST") |>
  req_body_multipart(images=form_file("images_few.zip", type="application/zip")) |>
  req_perform()

resp_body_json(r, simplifyVector=TRUE, simplifyDataFrame=FALSE, simplifyMatrix=TRUE)
# several (3 here) "predictions" elements


## Performance comparison ----

classify <- function(file) {
  file_extension <- tools::file_ext(file)
  if (file_extension == "zip") {
    type="application/zip"
  } else {
    type <- NULL
    # will be inferred
  }
  r <- request("https://inference-walton.cloud.imagine-ai.eu/system/services/zooprocess-multiple-classifier/exposed/main/v2/models/zooprocess_multiple_classifier/predict/") |>
    req_url_query(bottom_crop=31) |>
    req_method("POST") |>
    req_body_multipart(images=form_file(file, type=type)) |>
    req_perform()

  r <- resp_body_json(r, simplifyVector=TRUE)
  return(r)
}

system.time(all <- map(images, classify))
system.time(all <- classify("images.zip"))

# system.time(all <- classify("sample.zip"))
# # ~30s for 1330 images!
