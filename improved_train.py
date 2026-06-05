"""
Improved Training Script using VGG16 Transfer Learning
Creates improved_pneumonia_model.h5 with better accuracy
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np

# Dataset paths
train_dir = "archive/train"
val_dir = "archive/val"

# Enhanced data augmentation for better X-ray precision
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Slightly reduced for X-rays
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,  # X-rays can be flipped
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Handle brightness variations in X-rays
    channel_shift_range=0.1  # Slight color variations
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators (VGG16 expects 224x224)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")

# Compute class weights to handle class imbalance (0=NORMAL, 1=PNEUMONIA)
class_counts = np.bincount(train_generator.classes)
total = class_counts.sum()
class_weight = {
    0: float(total) / (2.0 * float(class_counts[0])),
    1: float(total) / (2.0 * float(class_counts[1])),
}
print(f"Class counts: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
print(f"Using class weights: {class_weight}")

# Load VGG16 base model (pre-trained on ImageNet)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('improved_pneumonia_model.h5', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train model (moderate epochs, with class weights)
print("\nTraining model for better X-ray precision...")
print("Using class weights for better Normal/Pneumonia balance.")
history = model.fit(
    train_generator,
    epochs=8,  # use more epochs for better learning
    # Let Keras infer steps_per_epoch so it sees the full dataset
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Fine-tuning: Unfreeze some layers and train with lower learning rate
print("\nFine-tuning model with class weights...")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=10,  # more epochs for fine-tuning
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save('improved_pneumonia_model.h5')
print("\nModel saved as improved_pneumonia_model.h5")

