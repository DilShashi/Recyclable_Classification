import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the final trained model
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

# Preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict the label for a single image
def predict_image(model, image_path, class_labels):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    return predicted_label, predictions[0][predicted_class]

# Display image with its predicted label
def display_image_with_label(image_path, predicted_label, confidence):
    img = load_img(image_path)  # Load image for display
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
    plt.show()

# List all real_world folders in the given path
def get_all_real_world_images(base_path):
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        if 'real_world' in root:  # Check for 'real_world' in the folder path
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for image files
                    image_paths.append(os.path.join(root, file))
    return image_paths

# Main function for real-time prediction
def main():
    model_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/final_trained_model.keras'
    base_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/raw/images/images'
    
    # Load the trained model
    model = load_trained_model(model_path)
    
    # Class labels (update as per your training labels)
    class_labels = ['Non-Recyclable', 'Recyclable']
    
    # Get all images from real_world folders
    image_paths = get_all_real_world_images(base_path)
    print(f"Found {len(image_paths)} images in real_world folders.")
    
    # Real-time prediction loop
    while True:
        print("\nEnter the full path of an image for prediction, or type 'exit' to quit:")
        input_path = input()
        
        if input_path.lower() == 'exit':
            print("Exiting real-time prediction...")
            break
        
        if os.path.exists(input_path) and input_path.lower().endswith(('png', 'jpg', 'jpeg')):
            # Predict and display the image with the label
            predicted_label, confidence = predict_image(model, input_path, class_labels)
            print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")
            display_image_with_label(input_path, predicted_label, confidence)
        else:
            print("Invalid image path or file type. Please try again.")

if __name__ == "__main__":
    main()
