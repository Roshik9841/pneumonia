"""
Evaluate Model Accuracy on Test Dataset
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Model to evaluate
model_path = "improved_pneumonia_model.h5"

if not os.path.exists(model_path):
    print("No model file found!")
    exit(1)

print(f"Loading model: {model_path}")
model = load_model(model_path)

# Determine input size based on model
if "improved" in model_path:
    img_size = (224, 224)
else:
    img_size = (150, 150)

# Test dataset
test_dir = "archive/test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"\nEvaluating on {test_generator.samples} test images...")

# Evaluate model
loss, accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Per-class accuracy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_generator.classes

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=['NORMAL', 'PNEUMONIA']))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

