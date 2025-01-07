import os
import random
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_path = "C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/raw/images"  # Path to the raw images folder
processed_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/processed/'  # Path to store processed datasets (train, valid, test)

# Create directories for processed images
train_dir = os.path.join(processed_path, 'train')
valid_dir = os.path.join(processed_path, 'valid')
test_dir = os.path.join(processed_path, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split dataset into train, validation, and test
def split_dataset(category, subfolder, images):
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
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to perform data augmentation
def augment_images(image_path, category, subfolder):
    img = Image.open(image_path)
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
    
    save_dir = './data/augmented/'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='png')):
        if i >= 20:  # Stop after generating 20 augmented images
            break

# Split dataset into categories, subfolders, and images
categories = os.listdir(dataset_path)

for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    if not os.path.isdir(category_path):
        continue
    
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Get list of images in subfolder
        images = os.listdir(subfolder_path)
        
        # Split images into train, valid, test
        train_images, valid_images, test_images = split_dataset(category, subfolder, images)
        
        # Create subfolder directories in train, valid, and test
        os.makedirs(os.path.join(train_dir, category, subfolder), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, category, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category, subfolder), exist_ok=True)

        # Copy images to respective directories
        for image in train_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(train_dir, category, subfolder, image))
        for image in valid_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(valid_dir, category, subfolder, image))
        for image in test_images:
            shutil.copy(os.path.join(subfolder_path, image), os.path.join(test_dir, category, subfolder, image))
        
        # Example of data preprocessing and augmentation 
        # Preprocess the first image from the train set as an example
        example_image_path = os.path.join(train_dir, category, subfolder, train_images[0])
        preprocessed_image = preprocess_image(example_image_path)
        print(f"Preprocessed image shape: {preprocessed_image.shape}")

        # Perform data augmentation on the first image
        augment_images(example_image_path, category, subfolder)

print("Dataset preparation complete: Split, Preprocessed, and Augmented!")
