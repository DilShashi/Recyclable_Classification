import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the final trained model
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

# Preprocess the image for prediction
def preprocess_image(img, target_size=(224, 224)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to match model input size
    img = img / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict class using the model
def predict_class(model, img):
    img_preprocessed = preprocess_image(img)
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions, axis=1)  # Get class with highest probability
    return predicted_class, predictions

# Display the prediction
def display_prediction(predicted_class, predictions):
    labels = ['Non-Recyclable', 'Recyclable']
    predicted_label = labels[predicted_class[0]]
    confidence = np.max(predictions) * 100  # Get confidence percentage
    print(f'Predicted: {predicted_label} with confidence: {confidence:.2f}%')

    # Display prediction on the image
    return predicted_label, confidence

# Main function for real-time webcam classification
def main():
    # Load the trained model
    model_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/final_trained_model.keras'
    model = load_trained_model(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Predict the class of the captured frame
        predicted_class, predictions = predict_class(model, frame)

        # Display the prediction on the frame
        predicted_label, confidence = display_prediction(predicted_class, predictions)

        # Show the image with the prediction
        cv2.putText(frame, f"{predicted_label}: {confidence:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-time Classification (Press q to quit)', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
