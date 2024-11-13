import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.windows
import matplotlib.pyplot as plt


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


def create_patches(sentinel_window_data, mask, patch_size=64, stride=32):
    """Generate patches of the sentinel data and mask."""
    sentinel_patches = []
    mask_patches = []

    for y in range(0, mask.shape[0] - patch_size, stride):
        for x in range(0, mask.shape[1] - patch_size, stride):
            sentinel_patch = sentinel_window_data[:, y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            sentinel_patches.append(sentinel_patch)
            mask_patches.append(mask_patch)

    sentinel_patches = np.stack(sentinel_patches)
    mask_patches = np.stack(mask_patches)
    return sentinel_patches, mask_patches


def label_patches(mask_patches):
    """Label the patches based on whether they contain park areas."""
    labels = np.zeros(len(mask_patches), dtype=np.uint8)
    for i, mask_patch in enumerate(mask_patches):
        if np.any(mask_patch):
            labels[i] = 1
    return labels


def save_patches(sentinel_patches, labels, output_folder, sentinel_data, window):
    """Save patches into two separate folders based on the label."""
    # If exists, remove the output folder and its contents
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
        os.rmdir(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "green"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "not_green"), exist_ok=True)

    for i, (sentinel_patch, label) in enumerate(zip(sentinel_patches, labels)):
        output_path = os.path.join(output_folder, f"{i}.tiff")
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=sentinel_patch.shape[1],
            width=sentinel_patch.shape[2],
            count=sentinel_patch.shape[0],
            dtype=sentinel_patch.dtype,
            crs=sentinel_data.crs,
            transform=sentinel_data.window_transform(window)
        ) as dst:
            dst.write(sentinel_patch)

        if label:
            os.rename(output_path, os.path.join(output_folder, "green", f"{i}.tiff"))
        else:
            os.rename(output_path, os.path.join(output_folder, "not_green", f"{i}.tiff"))


def visualize_random_patches(sentinel_patches, mask_patches, labels, n=10):
    """Visualize random patches with their labels."""
    indices = np.random.choice(len(sentinel_patches), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(15, 6))
    for i, index in enumerate(indices):
        rgb_patch = np.stack([
            sentinel_patches[index][0],  # Red band
            sentinel_patches[index][1],  # Green band
            sentinel_patches[index][2]   # Blue band
        ], axis=-1)

        axes[0, i].imshow(rgb_patch / 255)
        axes[0, i].set_title("Park" if labels[index] else "Non-park")
        axes[0, i].axis('off')

        axes[1, i].imshow(mask_patches[index], cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


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

    # Create patches and label them
    sentinel_patches, mask_patches = create_patches(sentinel_window_data, mask)
    labels = label_patches(mask_patches)

    # Save patches based on labels
    save_patches(sentinel_patches, labels, patches_output_folder, sentinel_data, window)

    # Visualize patches
    visualize_random_patches(sentinel_patches, mask_patches, labels)


# Run main function
if __name__ == "__main__":
    main()
