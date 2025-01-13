import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Function to build a classification model using a pretrained feature extractor
def build_classification_model(input_shape=(256, 256, 3), num_classes=2):
    # Using ResNet50 as a feature extractor (without the top layer)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze early layers and allow the later layers to be fine-tuned
    for layer in base_model.layers[:100]:  # Freeze the first 100 layers
        layer.trainable = False
    
    model = models.Sequential()
    model.add(base_model)
    
    # Add Batch Normalization to improve convergence
    model.add(layers.GlobalAveragePooling2D())  # Global Average Pooling
    model.add(layers.BatchNormalization())  # Batch Normalization before activation
    model.add(layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0005)))  # Slightly stronger regularization
    model.add(layers.ReLU())  # ReLU activation
    model.add(layers.Dropout(0.4))  # Lower dropout rate to reduce overfitting while still preventing it
    model.add(layers.BatchNormalization())  # Additional Batch Normalization
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer with softmax activation
    
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Low learning rate for fine-tuning
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Function to build an object detection model using bounding boxes
def build_object_detection_model(input_shape=(256, 256, 3), num_classes=2):
    inputs = layers.Input(shape=input_shape)
    
    # Feature extractor (simple CNN layers)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the features
    x = layers.Flatten()(x)

    # Bounding box regression (predicting coordinates)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)
    
    # Classification output
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss={'bbox_output': 'mean_squared_error', 'class_output': 'sparse_categorical_crossentropy'},
                  metrics={'bbox_output': 'mae', 'class_output': 'accuracy'})
    
    return model

if __name__ == "__main__":
    # Create classification model
    classification_model = build_classification_model(input_shape=(256, 256, 3), num_classes=2)
    classification_model.summary()
    
    # Implement callbacks for early stopping, learning rate reduction, and model checkpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
