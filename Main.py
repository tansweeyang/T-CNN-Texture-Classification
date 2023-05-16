import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import save_model, load_model
import tensorflow as tf

from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ImageLoader import ImageLoader

config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4)
session = tf.compat.v1.Session(config=config)

# Define path, image size and category
data_path = "dtd/"
img_size = 227
max_images_per_category = 1000

TRAIN_NETWORK = True

imageLoader = ImageLoader(data_path, img_size, max_images_per_category)
images, labels, categories = imageLoader.load_image()

# Show some statistics on the data
print("Number of categories:", len(categories))
print("Number of images:", len(images))
print("Shape of an image:", images[0].shape)

# Check the shape and type of the data and labels
print("Data shape:", images.shape)
print("Labels shape:", labels.shape)
print("Data type:", images.dtype)
print("Labels type:", labels.dtype)

# Create CNN model (T-CNN or LeNet5)
networkName = 'T-CNN'
cnn = ConvolutionalNeuralNetwork(networkName)
cnn.create_model_archiecture(categories)
cnn.build_model(images.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Convert the pixel values to floats
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# Scale the pixel values to the range [0, 1]
X_train /= 255.0
X_val /= 255.0
X_test /= 255.0

# Train CNN model
if not TRAIN_NETWORK:
    model, history = load_model(networkName + '.h5')
else:
    model, history = cnn.train_model(X_train, y_train, X_val, y_val)
    save_model(model, networkName + '.h5')

# Plot the training and validation curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()

plt.savefig('plot.png')

plt.show()
