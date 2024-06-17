import cv2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import DataProcessor

X_train, y_train, X_val, y_val = DataProcessor.train_test_data()

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

model = GaussianNB()
model.fit(X_train_flat, y_train)

y_pred = model.predict(X_val_flat)

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy on validation set:", accuracy)

conf_matrix = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_val, y_pred)
print("\nClassification Report:")
print(class_report)

folder_path_test = "test"
predicted_result = []

for filename in os.listdir(folder_path_test):
    img_path = os.path.join(folder_path_test, filename)
    if os.path.isfile(img_path):
        image = cv2.imread(img_path)
        resized_image = cv2.resize(image, (150, 150))
        flattened_image = resized_image.flatten()

        prediction = model.predict(flattened_image.reshape(1, -1))
        predicted_class_index = int(prediction)
        predicted_class = ["cat", "dog", "wild"][predicted_class_index - 1]

        predicted_result.append(predicted_class)

print("Predicted classes for test images:", predicted_result)
