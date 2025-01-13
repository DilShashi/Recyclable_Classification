import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the final trained model
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load the test data
def load_test_data(test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False  # Don't shuffle for evaluation to match predictions with true labels
    )

    return test_generator

# Evaluate the model
def evaluate_model(model, test_generator):
    # Get true labels and predictions
    true_labels = test_generator.classes
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate accuracy, precision, recall, F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1, true_labels, predicted_labels, predictions

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Recyclable', 'Recyclable'], yticklabels=['Non-Recyclable', 'Recyclable'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot ROC curve and calculate AUC
def plot_roc_curve(true_labels, predictions):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, predictions[:, 1])  # Using class 1 (Recyclable) for positive class
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Main entry point for evaluation
def main():
    # Path to the trained model and test data
    model_path = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/final_trained_model.keras'
    test_dir = 'C:/Users/dilan/Desktop/CV project/Recyclable_Classification/data/processed/test'

    # Load the trained model
    model = load_trained_model(model_path)

    # Load the test data
    test_generator = load_test_data(test_dir)

    # Evaluate the model
    accuracy, precision, recall, f1, true_labels, predicted_labels, predictions = evaluate_model(model, test_generator)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels)

    # Plot ROC curve
    plot_roc_curve(true_labels, predictions)

# Run the evaluation
if __name__ == "__main__":
    main()
