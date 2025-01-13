import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_classification_model
import os

# Load dataset with optimized data augmentation
def load_data(processed_data_path):
    train_dir = os.path.join(processed_data_path, 'train')
    valid_dir = os.path.join(processed_data_path, 'valid')

    # Increased data augmentation complexity for better model generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased rotation range to help the model generalize
        width_shift_range=0.2,  # Increased shift range for more variation
        height_shift_range=0.2,  # Increased shift range for more variation
        shear_range=0.2,  # Increased shear range
        zoom_range=0.2,  # Increased zoom range
        horizontal_flip=True,
        fill_mode='nearest'  # Ensures better image fill during transformations
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(224, 224),  # Image size is still optimized for memory
                                                        batch_size=32,
                                                        class_mode='sparse')
    
    valid_generator = valid_datagen.flow_from_directory(valid_dir,
                                                        target_size=(224, 224),  # Image size is still optimized for memory
                                                        batch_size=32,
                                                        class_mode='sparse')

    return train_generator, valid_generator

# Function to perform hyperparameter tuning with optimized learning rate and batch size
def hyperparameter_tuning(train_generator, valid_generator, learning_rate=0.001, batch_size=32, epochs=10):
    # Build the classification model
    model = build_classification_model(input_shape=(224, 224, 3), num_classes=2)

    # Set up early stopping, reduce learning rate on plateau, and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=valid_generator,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint],
                        batch_size=batch_size)
    
    return model, history

# Plot training history (accuracy and loss)
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='valid accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Run hyperparameter tuning
def tune_and_evaluate(processed_data_path):
    train_generator, valid_generator = load_data(processed_data_path)
    
    # Tune hyperparameters (learning rate, epochs)
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [16, 32, 64]  # Added more batch size options
    best_model = None
    best_history = None
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Tuning for lr={lr}, batch_size={batch_size}")
            model, history = hyperparameter_tuning(train_generator, valid_generator, learning_rate=lr, batch_size=batch_size, epochs=15)  # Increased epochs
            
            # Save the best model based on validation accuracy
            val_accuracy = max(history.history['val_accuracy'])
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_history = history
                best_params = {'learning_rate': lr, 'batch_size': batch_size}

    print(f"Best model achieved validation accuracy of: {best_accuracy}")
    print(f"Best hyperparameters: {best_params}")

    # Plot history of the best model
    plot_history(best_history)

    # Save the best model explicitly
    best_model.save("best_model.keras")

    return best_params

if __name__ == "__main__":
    processed_data_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/processed'
    best_params = tune_and_evaluate(processed_data_path)
