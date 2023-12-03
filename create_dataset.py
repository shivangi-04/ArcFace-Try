import os


DIRNAME = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\Data\Images'


import os
import shutil
import random

# Set the paths
main_folder = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\Data\Images'
train_folder = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\Train'
validation_folder = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\Validation'
test_folder = r'C:\Users\Dell\Documents\cfp-dataset\cfp-dataset\Test'

# Create Train, Validation, and Test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Iterate through subfolders in the main folder
for sub_folder_name in os.listdir(main_folder):
    sub_folder_path = os.path.join(main_folder, sub_folder_name)

    # Ensure it's a directory and not a file
    if os.path.isdir(sub_folder_path):
        # List the images in the Frontal and Profile folders
        frontal_images = os.listdir(os.path.join(sub_folder_path, 'Frontal'))
        frontal_images = [os.path.join('Frontal', x) for x in frontal_images]
        profile_images = os.listdir(os.path.join(sub_folder_path, 'Profile'))
        profile_images = [os.path.join('Profile', x) for x in profile_images]

        # Randomly select 12 images for training, 1 for validation, and 1 for testing
        selected_images = random.sample(frontal_images + profile_images, 14)


        # Create subfolders in Train, Validation, and Test folders
        train_sub_folder = os.path.join(train_folder, sub_folder_name)
        validation_sub_folder = os.path.join(validation_folder, sub_folder_name)
        test_sub_folder = os.path.join(test_folder, sub_folder_name)

        os.makedirs(train_sub_folder, exist_ok=True)
        os.makedirs(validation_sub_folder, exist_ok=True)
        os.makedirs(test_sub_folder, exist_ok=True)

        # Copy selected images to Train, Validation, and Test subfolders
        for idx, img in enumerate(selected_images[:12]):
            src_path = os.path.join(sub_folder_path, img)
            dest_path = os.path.join(train_sub_folder, f'{str(idx + 1).zfill(2)}.jpg')
            shutil.copy(src_path, dest_path)

        for idx, img in enumerate(selected_images[-2 : -1]):
            src_path = os.path.join(sub_folder_path, img)
            dest_path = os.path.join(validation_sub_folder, f'{str(idx + 1).zfill(2)}.jpg')
            shutil.copy(src_path, dest_path)

        for idx, img in enumerate(selected_images[-1 : ]):
            src_path = os.path.join(sub_folder_path, img)
            dest_path = os.path.join(test_sub_folder, f'{str(idx + 1).zfill(2)}.jpg')
            shutil.copy(src_path, dest_path)
