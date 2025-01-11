import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# File paths
test_dir = 'C:\\Users\\dilan\\Desktop\\CV project\\Recyclable_Classification\\data\\processed\\test'

# Load the trained model
model = load_model('classification_model.h5')

# Data preprocessing for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Data loading for test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',  # 'binary' for binary classification
    shuffle=False  # Do not shuffle for evaluation
)

# Predict on the test set
y_pred = model.predict(test_generator)
y_pred_class = (y_pred > 0.5).astype("int32")  # Threshold to classify as 0 or 1

# Get true labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_class)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred_class)
print("Classification Report:")
print(report)

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Accuracy, Precision, Recall, F1 Score
accuracy = np.mean(y_pred_class == y_true)
print(f"Accuracy: {accuracy:.4f}")
