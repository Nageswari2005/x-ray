import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Fix typo: "rescle" → "rescale"
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# 2. Use correct relative paths (as shown in your folder structure)
train = train_datagen.flow_from_directory(
    "archive (5)/chest_xray/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test = test_datagen.flow_from_directory(
    "archive (5)/chest_xray/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 3. Fix incorrect layer references (use tf.keras.layers not tf.keras.layer)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=test, epochs=5)

# 5. Print model summary
model.summary()
model.save("pneumonia_model.h5")