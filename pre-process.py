import os


def split_test_train_val():
    path = "data/parks_urban_data/"
    class1 = "green"
    class2 = "not_green"

    # split the data into train, test and validation
    class1_folders = os.listdir(path + class1)
    class2_folders = os.listdir(path + class2)

    # Create the folders
    os.mkdir(path + "train")
    os.mkdir(path + "test")
    os.mkdir(path + "val")

    os.mkdir(path + "train/" + class1)
    os.mkdir(path + "train/" + class2)

    os.mkdir(path + "test/" + class1)
    os.mkdir(path + "test/" + class2)

    os.mkdir(path + "val/" + class1)
    os.mkdir(path + "val/" + class2)

    # Split the data
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








# Read tif file

def unload_folders():

    path = "data/parks_urban_data/"
    class1 = "green"
    class2 = "not_green"

    # Extract all folders in both class paths
    class1_folders = os.listdir(path + class1)
    class2_folders = os.listdir(path + class2)

    counter = 0
    for folder in class1_folders:
        # Extract all tif files in the folder
        tif_files = os.listdir(path + class1 + "/" + folder)
        for tif in tif_files:
            # move tif file to the class1 folder
            os.rename(path + class1 + "/" + folder + "/" + tif, path + class1 + "/" + str(counter) + ".jpg")
            counter += 1

    counter = 0
    for folder in class2_folders:
        # Extract all tif files in the folder
        tif_files = os.listdir(path + class2 + "/" + folder)
        for tif in tif_files:
            # move tif file to the class2 folder
            os.rename(path + class2 + "/" + folder + "/" + tif, path + class2 + "/" + str(counter) + ".jpg")
            counter += 1


    # Remove empty folders
    for folder in class1_folders:
        os.rmdir(path + class1 + "/" + folder)

    for folder in class2_folders:
        os.rmdir(path + class2 + "/" + folder)


def pre_process_sentinal_data():
    from PIL import Image
    import numpy as np

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

    if not os.path.exists('data/sent-2/patches'):
        os.makedirs('data/sent-2/patches')
    else:
        for file in os.listdir('data/sent-2/patches'):
            os.remove(f'data/sent-2/patches/{file}')

    for i, patch in enumerate(patches):
        Image.fromarray(patch).save(f'data/sent-2/patches/patch_{i}.jpg')


def resize_images():
    import cv2
    import os

    folder = "data/parks_urban_data/val/green"
    new_folder = "data/resized_parks_urban_data/val/green"
    # remove the data if exists
    if os.path.exists(new_folder):
        for file in os.listdir(new_folder):
            os.remove(os.path.join(new_folder, file))
    else:
        os.makedirs(new_folder)


    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (128, 128))
        cv2.imwrite(os.path.join(new_folder, filename), img)



resize_images()