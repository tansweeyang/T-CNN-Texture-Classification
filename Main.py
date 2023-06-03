import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import save_model, load_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import seaborn as sns
import pickle

from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ImageLoader import ImageLoader

config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4)
session = tf.compat.v1.Session(config=config)

# Define settings
TRAIN_NETWORK = True
dataset_path = "dtd/"
img_size = 227
networkName = 'T-CNN'  # T-CNN or ResNet

imageLoader = ImageLoader(dataset_path, img_size)

for i in range(10):
    train_file_content = imageLoader.readSplitFileContent('train' + str(i + 1))
    val_file_content = imageLoader.readSplitFileContent('val' + str(i + 1))
    test_file_content = imageLoader.readSplitFileContent('test' + str(i + 1))

    X_train, y_train = imageLoader.load_predefined_split_images(train_file_content)
    X_val, y_val = imageLoader.load_predefined_split_images(val_file_content)
    X_test, y_test = imageLoader.load_predefined_split_images(test_file_content)

    train_images = imageLoader.preprocessImages(train_images)
    val_images = imageLoader.preprocessImages(val_images)
    test_images = imageLoader.preprocessImages(test_images)

    # Show some statistics on the data
    print("Number of categories:", len(imageLoader.category))
    print("Number of images:", len(imageLoader.images))
    print("Shape of an image:", train_images[0].shape)
    print("Labels shape:", X_train[0].shape)
    print("Data type:", X_train[0].dtype)
    print("Labels type:", X_train[0].dtype)

    # Create CNN mode
    cnn = ConvolutionalNeuralNetwork(networkName)
    cnn.create_model_archiecture(len(imageLoader.category))
    cnn.build_model(imageLoader.images.shape)

    X_train = imageLoader.normalizeImages(X_train)
    X_val = imageLoader.normalizeImages(X_val)
    X_test = imageLoader.normalizeImages(X_test)

# Train CNN model
if not TRAIN_NETWORK:
    model = load_model(networkName + '.h5')
    with open('history.pkl', 'rb') as f:
        history = pickle.load(f)  # Load the history object
else:
    model, history = cnn.train_model(X_train, y_train, X_val, y_val)
    save_model(model, networkName + '.h5')
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)  # Save the history object

# Make predictions using the trained model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute and generate confusion matrix, classification report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
accuracy = accuracy_score(y_true, y_pred)

print(report)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Accuracy:", accuracy)

# To generate txt file with classification report copied
with open('classification_report.txt', 'w') as f:
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'F1-Score: {f1:.4f}\n\n')
    f.write(report)
    f.write('\n')
    f.write(f'Accuracy: {accuracy:.4f}\n\n')

# To generate visualization of confusion matrix and save it
fig, ax = plt.subplots(figsize=((2.5*len(categories)), (2.5*len(categories))))
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Lenet-5 Model Confusion Matrix')
ax.xaxis.set_ticklabels(categories)
ax.yaxis.set_ticklabels(categories)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('Confusion Matrix.png')
plt.show()
plt.clf()

# Get the accuracy, validation accuracy, loss, and validation loss values from the history object
if not TRAIN_NETWORK:
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
else:
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

# Create line graph for validation loss and loss
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metrics', fontsize=12)
plt.title('Lenet-5 Model Loss', fontsize=14)
plt.legend(fontsize=10)
plt.savefig('Val Loss VS Loss Curves.png')
plt.clf()

# Create line graph for validation acc and acc
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metrics', fontsize=12)
plt.title('Lenet-5 Model Accuracy', fontsize=14)
plt.legend(fontsize=10)
plt.savefig('Val Acc VS Acc Curves.png')
plt.clf()

# Create bar graph for validation accuracy and accuracy
plt.bar(['Validation Accuracy', 'Training Accuracy'], [val_accuracy[-1], accuracy[-1]])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Lenet-5 Validation Accuracy vs Training Accuracy', fontsize=14)
plt.savefig('Val Acc VS Acc Bar.png')
plt.clf()

# Create bar graph for validation loss and loss
plt.bar(['Validation Loss', 'Training Loss'], [val_loss[-1], loss[-1]])
plt.ylabel('Loss', fontsize=12)
plt.title('Lenet-5 Validation Loss vs Training Loss', fontsize=14)
plt.savefig('Val Loss VS Loss Bar.png')
plt.clf()