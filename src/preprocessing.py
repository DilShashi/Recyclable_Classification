import os
import random
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import kaggle
from collections import Counter
import matplotlib.pyplot as plt

# Kaggle dataset details
kaggle_dataset_url = 'alistairking/recyclable-and-household-waste-classification'
dataset_zip = 'recyclable-and-household-waste-classification.zip'
dataset_dir = r'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/raw'

# Ensure the Kaggle API key is set up
if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
    raise Exception("Kaggle API key is not set up. Please set it up before running this script.")

# Download the dataset if not already downloaded
dataset_zip_path = os.path.join(dataset_dir, dataset_zip)
if not os.path.exists(dataset_zip_path):
    print("Downloading dataset...")
    kaggle.api.dataset_download_files(kaggle_dataset_url, path=dataset_dir, unzip=False)
    print("Download complete.")

# Extract the dataset if not already extracted
data_extracted_path = os.path.join(dataset_dir, 'images')
if not os.path.exists(data_extracted_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction complete.")

# Define paths
dataset_path = os.path.join(dataset_dir, 'images', 'images')  # Path to the images folder
processed_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/processed/'  # Path to store processed datasets (train, valid, test)

# Create directories for processed images
train_dir = os.path.join(processed_path, 'train')
valid_dir = os.path.join(processed_path, 'valid')
test_dir = os.path.join(processed_path, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split dataset into train, validation, and test
def split_dataset(category, images):
    random.shuffle(images)
    num_images = len(images)
    num_train = int(0.7 * num_images)
    num_valid = int(0.15 * num_images)
    num_test = num_images - num_train - num_valid

    # Split images
    train_images = images[:num_train]
    valid_images = images[num_train:num_train + num_valid]
    test_images = images[num_train + num_valid:]

    return train_images, valid_images, test_images

# Function to preprocess image (resize and normalize)
def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to perform data augmentation
def augment_images(image_path, save_dir):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).reshape((1,) + img.size + (3,))

    # Augment and save images
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png')):
        if i >= 20:  # Stop after generating 20 augmented images
            break

# Relabel categories into Recyclable and Non-Recyclable
def relabel_category(category):
    recyclable = [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
        'cardboard_boxes', 'cardboard_packaging', 'glass_beverage_bottles',
        'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
        'newspaper', 'office_paper', 'plastic_detergent_bottles',
        'plastic_food_containers', 'plastic_soda_bottles',
        'plastic_water_bottles', 'steel_food_cans'
    ]
    return 'Recyclable' if category in recyclable else 'Non-Recyclable'

# Check class distribution
def check_class_distribution(categories):
    distribution = Counter(categories)
    plt.bar(distribution.keys(), distribution.values())
    plt.title("Class Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.show()

# Process the dataset
categories = os.listdir(dataset_path)
all_images = []

for category in categories:
    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    relabeled_category = relabel_category(category)

    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        # Get list of images in subfolder
        images = os.listdir(subfolder_path)

        # Add to overall image list for class distribution
        all_images.extend([relabeled_category] * len(images))

        # Split images into train, valid, test
        train_images, valid_images, test_images = split_dataset(relabeled_category, images)

        # Create subfolder directories in train, valid, and test
        os.makedirs(os.path.join(train_dir, relabeled_category), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, relabeled_category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, relabeled_category), exist_ok=True)

        # Copy images to respective directories
        for image in train_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(train_dir, relabeled_category, image))
        for image in valid_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(valid_dir, relabeled_category, image))
        for image in test_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(test_dir, relabeled_category, image))

        # Example of data preprocessing and augmentation 
        # Preprocess the first image from the train set as an example
        example_image_path = os.path.join(train_dir, relabeled_category, train_images[0])
        preprocessed_image = preprocess_image(example_image_path, target_size=(224, 224))
        print(f"Preprocessed image shape: {preprocessed_image.shape}")

        # Perform data augmentation on the first image
        augment_dir = os.path.join(processed_path, 'augmented', relabeled_category)
        os.makedirs(augment_dir, exist_ok=True)
        augment_images(example_image_path, augment_dir)

# Check and plot class distribution
check_class_distribution(all_images)

print("Dataset preparation complete: Split, Preprocessed, Augmented, and Relabeled!")
