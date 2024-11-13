import os
import cv2
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.windows
import matplotlib.pyplot as plt
from PIL import Image



def load_data(park_path, sentinel_path):
    """Load park and sentinel data."""
    park_data = gpd.read_file(park_path)
    sentinel_data = rasterio.open(sentinel_path)
    return park_data, sentinel_data


def calculate_intersection_bounds(park_data, sentinel_data):
    """Calculate intersection bounding box of park and sentinel data."""
    minx, miny, maxx, maxy = park_data.total_bounds
    sentinel_bounds = sentinel_data.bounds
    minx = max(minx, sentinel_bounds.left)
    miny = max(miny, sentinel_bounds.bottom)
    maxx = min(maxx, sentinel_bounds.right)
    maxy = min(maxy, sentinel_bounds.top)
    return minx, miny, maxx, maxy


def create_mask(park_data, sentinel_data, bounds):
    """Create a mask for the park area within the bounds."""
    minx, miny, maxx, maxy = bounds
    window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, sentinel_data.transform)
    sentinel_window_data = sentinel_data.read(window=window)

    mask = np.zeros((sentinel_window_data.shape[1], sentinel_window_data.shape[2]), dtype=np.uint8)
    mask = rasterio.features.rasterize(
        [(geometry, 1) for geometry in park_data.geometry],
        out_shape=(sentinel_window_data.shape[1], sentinel_window_data.shape[2]),
        transform=sentinel_data.window_transform(window),
        fill=0,
        dtype=np.uint8
    )
    return mask, sentinel_window_data, window


def save_mask_as_tiff(mask, sentinel_data, window, output_path):
    """Save the mask array as a GeoTIFF file."""
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=sentinel_data.crs,
        transform=sentinel_data.window_transform(window)
    ) as dst:
        dst.write(mask, 1)


def split_test_train_val():
    path = "data/Seattle_Parks/patches/"
    class1 = "green"
    class2 = "not_green"

    class1_folders = os.listdir(path + class1)
    class2_folders = os.listdir(path + class2)

    os.makedirs(path + "train/" + class1, exist_ok=True)
    os.makedirs(path + "train/" + class2, exist_ok=True)
    os.makedirs(path + "test/" + class1, exist_ok=True)
    os.makedirs(path + "test/" + class2, exist_ok=True)
    os.makedirs(path + "val/" + class1, exist_ok=True)
    os.makedirs(path + "val/" + class2, exist_ok=True)

    for i in range(len(class1_folders)):
        if i < 0.7 * len(class1_folders):
            os.rename(path + class1 + "/" + class1_folders[i], path + "train/" + class1 + "/" + class1_folders[i])
        elif i < 0.85 * len(class1_folders):
            os.rename(path + class1 + "/" + class1_folders[i], path + "test/" + class1 + "/" + class1_folders[i])
        else:
            os.rename(path + class1 + "/" + class1_folders[i], path + "val/" + class1 + "/" + class1_folders[i])

    for i in range(len(class2_folders)):
        if i < 0.7 * len(class2_folders):
            os.rename(path + class2 + "/" + class2_folders[i], path + "train/" + class2 + "/" + class2_folders[i])
        elif i < 0.85 * len(class2_folders):
            os.rename(path + class2 + "/" + class2_folders[i], path + "test/" + class2 + "/" + class2_folders[i])
        else:
            os.rename(path + class2 + "/" + class2_folders[i], path + "val/" + class2 + "/" + class2_folders[i])

    # Remove the original folders
    os.rmdir(path + class1)
    os.rmdir(path + class2)


def unload_folders():
    path = "data/parks_urban_data/"
    class1 = "green"
    class2 = "not_green"

    # If exists, remove the output folder and its contents
    if os.path.exists(path + class1):
        for file in os.listdir(path + class1):
            os.remove(os.path.join(path + class1, file))
        os.rmdir(path + class1)

    class1_folders = os.listdir(path + class1)
    class2_folders = os.listdir(path + class2)

    counter = 0
    for folder in class1_folders:
        tif_files = os.listdir(path + class1 + "/" + folder)
        for tif in tif_files:
            os.rename(path + class1 + "/" + folder + "/" + tif, path + class1 + "/" + str(counter) + ".jpg")
            counter += 1

    counter = 0
    for folder in class2_folders:
        tif_files = os.listdir(path + class2 + "/" + folder)
        for tif in tif_files:
            os.rename(path + class2 + "/" + folder + "/" + tif, path + class2 + "/" + str(counter) + ".jpg")
            counter += 1

    for folder in class1_folders:
        os.rmdir(path + class1 + "/" + folder)

    for folder in class2_folders:
        os.rmdir(path + class2 + "/" + folder)


def pre_process_sentinal_data():
    def segment_image(image, patch_size=(256, 256)):
        patches = []
        image_array = np.array(image)
        for y in range(0, image_array.shape[0], patch_size[1]):
            for x in range(0, image_array.shape[1], patch_size[0]):
                patch = image_array[y:y+patch_size[1], x:x+patch_size[0]]
                if patch.shape[:2] == patch_size:
                    patches.append(patch)
        return patches

    image = Image.open('data/sent-2/2024-05-14-00_00_2024-05-14-23_59_Sentinel-2_L2A_True_Color.jpg')
    patches = segment_image(image)

    os.makedirs('data/sent-2/patches', exist_ok=True)
    for file in os.listdir('data/sent-2/patches'):
        os.remove(f'data/sent-2/patches/{file}')

    for i, patch in enumerate(patches):
        Image.fromarray(patch).save(f'data/sent-2/patches/patch_{i}.jpg')



# Main execution flow
def main():
    # Paths to data
    park_data_path = "data/Seattle_Parks/park_mask.geojson"
    sentinel_data_path = "data/Seattle_Parks/sentinel_data.tiff"
    mask_output_path = "data/Seattle_Parks/park_mask.tiff"
    patches_output_folder = "data/Seattle_Parks/patches"

    # Load data
    park_data, sentinel_data = load_data(park_data_path, sentinel_data_path)

    # Calculate intersection bounds and create mask
    bounds = calculate_intersection_bounds(park_data, sentinel_data)
    mask, sentinel_window_data, window = create_mask(park_data, sentinel_data, bounds)

    # Save mask
    save_mask_as_tiff(mask, sentinel_data, window, mask_output_path)

    # Additional processing functions
    split_test_train_val()


# Run main function
if __name__ == "__main__":
    main()
