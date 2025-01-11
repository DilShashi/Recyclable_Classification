import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_classification_model, compile_model

# File paths
train_dir = 'C:\\Users\\dilan\\Desktop\\CV project\\Recyclable_Classification\\data\\processed\\train'
valid_dir = 'C:\\Users\\dilan\\Desktop\\CV project\\Recyclable_Classification\\data\\processed\\valid'

# Ensure that the directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")

if not os.path.exists(valid_dir):
    raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

# Define input shape and batch size
input_shape = (256, 256, 3)
batch_size = 32

# 4.1 Define loss function, optimizer, and data augmentation
model = create_classification_model(input_shape=input_shape)

# Modify the compile model function to avoid object detection loss
model.compile(
    loss='binary_crossentropy',  # Use binary crossentropy for binary classification
    optimizer='adam',
    metrics=['accuracy']
)

# Data augmentation and data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Data loading
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'  # 'binary' for binary classification
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary'  # 'binary' for binary classification
)

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Saving the trained model
model.save('classification_model.h5')
