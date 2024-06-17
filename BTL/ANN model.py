import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import DataProcessor

num_classes = 3

model = models.Sequential([
    layers.Flatten(input_shape=(225,)),
    layers.Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, y_train, X_val, y_val = DataProcessor.train_test_data()
X_train = X_train / 255.0
X_val = X_val / 255.0

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1))

pca = PCA(n_components=225)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

history = model.fit(X_train_pca, y_train, epochs=20, validation_data=(X_val_pca, y_val))

val_loss, val_accuracy = model.evaluate(X_val_pca, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

y_pred = np.argmax(model.predict(X_val_pca), axis=1)

conf_matrix = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_val, y_pred)
print("Classification Report:")
print(class_report)

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

folder_path_test = "test"
predicted_result = []

for filename in os.listdir(folder_path_test):
    img_path = os.path.join(folder_path_test, filename)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, (150, 150))
        normalized_image = resized_image / 255.0
        image_scaled = scaler.transform(normalized_image.flatten().reshape(1, -1))

        image_pca = pca.transform(image_scaled)

        prediction = model.predict(image_pca.reshape(1, -1))
        predicted_class_index = np.argmax(prediction)
        predicted_class = ["cat", "dog", "wild"][predicted_class_index]

        predicted_result.append(predicted_class)

print("Predicted class:", predicted_result)
