from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import confusion_matrix, classification_report
import DataProcessor
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

input_shape = (150, 150, 3)
num_classes = 3

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, y_train, X_val, y_val = DataProcessor.train_test_data()
X_train = X_train / 255.0
X_val = X_val / 255.0

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_val, y_pred_labels)

class_report = classification_report(y_val, y_pred_labels)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

folder_path_test = "test"
predicted_result = []

for filename in os.listdir(folder_path_test):
    img_path = os.path.join(folder_path_test, filename)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, (150, 150))
        normalized_image = resized_image / 255.0

        prediction = model.predict(np.expand_dims(normalized_image, axis=0))
        predicted_class_index = np.argmax(prediction)
        predicted_class = ["cat", "dog", "wild"][predicted_class_index]

        predicted_result.append(predicted_class)

print("Predicted class:", predicted_result)
