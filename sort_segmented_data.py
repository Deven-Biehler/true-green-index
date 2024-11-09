# Segment the sentinal data and park mask into patches
# Compare the coordinates of the patches with the ground truth image
# Classify the image as positive or negative

import os
import numpy as np
import geopandas as gpd


Park_data = "data/Seattle_Parks/park_mask.geojson"
Sentinel_data = "data/Seattle_Parks/sentinel_data.tiff"


# Load the park data
park_data = gpd.read_file(Park_data)
park_data = park_data.to_crs(epsg=32610)