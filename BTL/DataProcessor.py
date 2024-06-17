import os
import numpy as np
import cv2

def read_and_label_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            images.append(img)
            labels.append(label)
    return images, labels

def train_test_data():
    folder_path_train_cat = "data/train_resized/cat"
    folder_path_train_dog = "data/train_resized/dog"
    folder_path_train_wild = "data/train_resized/wild"

    folder_path_val_cat = "data/val_resized/cat"
    folder_path_val_dog = "data/val_resized/dog"
    folder_path_val_wild = "data/val_resized/wild"

    X_train, y_train, X_val, y_val = [], [], [], []

    folder_paths_train = [folder_path_train_cat, folder_path_train_dog, folder_path_train_wild]
    for folder_path, label in zip(folder_paths_train, [0, 1, 2]):
        images, labels = read_and_label_images(folder_path, label)
        X_train.extend(images)
        y_train.extend(labels)

    folder_paths_val = [folder_path_val_cat, folder_path_val_dog, folder_path_val_wild]
    for folder_path, label in zip(folder_paths_val, [0, 1, 2]):
        images, labels = read_and_label_images(folder_path, label)
        X_val.extend(images)
        y_val.extend(labels)

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val)
    y_val = np.array(y_val, dtype=np.int32)

    train_indices = np.arange(X_train.shape[0])
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    return X_train, y_train, X_val, y_val