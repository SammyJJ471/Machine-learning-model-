#Import necessary libraries
import tensorflow as tf from tensorflow
import keras from sklea.model_selection
import train_test_split from sklearn.metrics import accuracy_score
import numpy as np
import os
import cv2

#Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#Define constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10

#Load and preprocess datasets
def load_dataset(dataset_dir):
    dataset = []
    labels = []
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            dataset.append(img)
            labels.append(label)
    return np.array(dataset), np.array(labels)

#Load fruits dataset
fruits_dataset_dir = 'path/to/fruits/dataset'
fruits_dataset, fruits_labels = load_dataset(fruits_dataset_dir)

#Load dogs dataset
dogs_dataset_dir = 'path/to/dogs/dataset'
dogs_dataset, dogs_labels = load_dataset(dogs_dataset_dir)

#Combine datasets
dataset = np.concatenate((fruits_dataset, dogs_dataset))
labels = np.concatenate((fruits_labels, dogs_labels))

#One-hot encode labels
unique_labels = np.unique(labels)
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
labels_onehot = np.zeros((len(labels), len(unique_labels)))
for i, label in enumerate(labels):
    labels_onehot[i, label_to_idx[label]] = 1

#Split dataset into training and validation sets
train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels_onehot, test_size=0.2, random_state=42)

#Normalize pixel values
train_dataset = train_dataset.astype('float32') / 255.0
val_dataset = val_dataset.astype('float32') / 255.0

#Define model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(unique_labels), activation='softmax')
])

#Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train model
history = model.fit(train_dataset, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_dataset, val_labels))

#Evaluate model
test_loss, test_acc = model.evaluate(val_dataset, val_labels)
print(f'Test accuracy: {test_acc:.2f}')

#Use model for predictions
predictions = model.predict(val_dataset)
predicted_labels = np.argmax(predictions, axis=1)

Convert predicted labels back to original label names
predicted_labels = [unique_labels[label] for label in predicted_labels]

#Print predicted labels
print(predicted_labels)