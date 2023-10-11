import time

from keras.models import save_model
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

from Plotter import Plotter
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ImageLoader import ImageLoader

config = tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=4,
    intra_op_parallelism_threads=4)
session = tf.compat.v1.Session(config=config)

# Settings
DATASET_PATH = 'dtd/'
# Set to true of first run
LOAD_DATA = False
TRAIN_NETWORK = False
# Set to true after training both network
COMPARE_NETWORKS = True

# Network Settings
NETWORK_NAME = 'LeNet5'  # T-CNN / LeNet5
network_image_sizes = {'T-CNN': 227, 'LeNet5': 32}
IMAGE_SIZE = network_image_sizes[NETWORK_NAME]
K_FOLDS = 10
EPOCHS = 35

# Create Image Loader
imageLoader = ImageLoader(DATASET_PATH)

accuracy_list = []
val_accuracy_list = []
loss_list = []
val_loss_list = []
final_accuracy_list = []

# Output: Saved X and y for all folds
if LOAD_DATA:
    print('\nBegin image loading and preprocessing...')
    start = time.time()

    for i in range(K_FOLDS):
        print('Fold (' + str(i + 1) + '/' + str(K_FOLDS) + ')')

        # Read cv file content
        train_file_content, val_file_content, test_file_content = imageLoader.read_cv_content('train' + str(i + 1),
                                                                                              'val' + str(i + 1),
                                                                                              'test' + str(i + 1))

        # Load X and y from content
        X_train, y_train = imageLoader.load_predefined_split_images(train_file_content, IMAGE_SIZE)
        X_val, y_val = imageLoader.load_predefined_split_images(val_file_content, IMAGE_SIZE)
        X_test, y_test = imageLoader.load_predefined_split_images(test_file_content, IMAGE_SIZE)

        # Preprocess images
        X_train = ImageLoader.preprocess_images(X_train)
        X_val = ImageLoader.preprocess_images(X_val)
        X_test = ImageLoader.preprocess_images(X_test)
        X_train = ImageLoader.normalize_images(X_train)
        X_val = ImageLoader.normalize_images(X_val)
        X_test = ImageLoader.normalize_images(X_test)

        # Display loaded image information
        print("Number of categories:", len(imageLoader.categories))
        with open('len_categories_loaded' + '.npy', 'wb') as f:
            np.save(f, len(imageLoader.categories))
        print("Number of images:", len(X_train))
        print("Shape of an image:", X_train[0].shape)
        print("Labels shape:", y_train[0].shape)
        print("Data type:", X_train[0].dtype)
        print("Labels type:", X_train[0].dtype)

        # Save X and y
        with open('train_test_split/' + NETWORK_NAME + '_X-train_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, X_train)
        with open('train_test_split/' + NETWORK_NAME + '_y-train_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, y_train)
        with open('train_test_split/' + NETWORK_NAME + '_X-val_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, X_val)
        with open('train_test_split/' + NETWORK_NAME + '_y-val_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, y_val)
        with open('train_test_split/' + NETWORK_NAME + '_X-test_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, X_test)
        with open('train_test_split/' + NETWORK_NAME + '_y-test_' + str(i+1) + '.npy', 'wb') as f:
            np.save(f, y_test)

    end = time.time()
    load_time = np.round((end - start) / 60, 2)
    print('Image loading and preprocessing completed.')
    print('Image loading and preprocessing time: ' + str(load_time) + 'min(s)')

# Input: Saved X and y for all folds
# Output: accuracy_list, val_accuracy_list, loss_list, val_loss_list, average_final_acc, final_std_accuracy
if TRAIN_NETWORK:
    print('\nBegin CNN training...')
    start = time.time()

    # For every folds
    for i in range(K_FOLDS):
        # Load train test split
        with open('train_test_split/' + NETWORK_NAME + '_X-train_' + str(i + 1) + '.npy', 'rb') as f:
            X_train = np.load(f)
        with open('train_test_split/' + NETWORK_NAME + '_y-train_' + str(i + 1) + '.npy', 'rb') as f:
            y_train = np.load(f)
        with open('train_test_split/' + NETWORK_NAME + '_X-val_' + str(i + 1) + '.npy', 'rb') as f:
            X_val = np.load(f)
        with open('train_test_split/' + NETWORK_NAME + '_y-val_' + str(i + 1) + '.npy', 'rb') as f:
            y_val = np.load(f)
        with open('train_test_split/' + NETWORK_NAME + '_X-test_' + str(i + 1) + '.npy', 'rb') as f:
            X_test = np.load(f)
        with open('train_test_split/' + NETWORK_NAME + '_y-test_' + str(i + 1) + '.npy', 'rb') as f:
            y_test = np.load(f)

        # Create CNN model
        cnn = ConvolutionalNeuralNetwork()
        with open('len_categories_loaded' + '.npy', 'rb') as f:
            len_categories = np.load(f)
        cnn.create_model_architecture(NETWORK_NAME, len_categories)
        cnn.build_model(X_train.shape)

        # Train CNN model
        print('Fold (' + str(i + 1) + '/' + str(K_FOLDS) + ')')

        checkpoint_path = 'checkpoints/' + NETWORK_NAME + '_checkpoint_' + str(i+1) + '.ckpt'
        model, history = cnn.train_model(X_train, y_train, X_val, y_val, EPOCHS, checkpoint_path)
        model.load_weights(checkpoint_path)

        # Add history to list
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        accuracy_list.append(accuracy)
        val_accuracy_list.append(val_accuracy)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        loss_list.append(loss)
        val_loss_list.append(val_loss)

        # Make predictions using the trained model
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Get final accuracy
        final_accuracy = accuracy_score(y_true, y_pred)
        final_accuracy_list.append(final_accuracy)

    average_final_acc = round(np.average(final_accuracy_list), 2)
    std_dev_final_acc = round(np.std(final_accuracy_list), 2)

    # Save accuracy_list, val_accuracy_list, loss_list, val_loss_list, average_final_acc, final_std_accuracy,
    # model parameters, model training history
    # accuracy_list, val_accuracy_list, loss_list, val_loss_list
    with open('training_acc_loss_lists/' + NETWORK_NAME + '_accuracy_list' + '.npy', 'wb') as f:
        np.save(f, accuracy_list)
    with open('training_acc_loss_lists/' + NETWORK_NAME + '_val_accuracy_list' + '.npy', 'wb') as f:
        np.save(f, val_accuracy_list)
    with open('training_acc_loss_lists/' + NETWORK_NAME + '_loss_list' + '.npy', 'wb') as f:
        np.save(f, loss_list)
    with open('training_acc_loss_lists/' + NETWORK_NAME + '_val_loss_list' + '.npy', 'wb') as f:
        np.save(f, val_loss_list)

    # Save average_final_acc, final_std_accuracy
    with open('final_acc_std/' + NETWORK_NAME + '_final_avg_accuracy' + '.npy', 'wb') as f:
        np.save(f, average_final_acc)
    with open('final_acc_std/' + NETWORK_NAME + '_final_std_accuracy' + '.npy', 'wb') as f:
        np.save(f, std_dev_final_acc)

    # Save CNN parameters
    save_model(model, 'models/' + NETWORK_NAME + '.h5')

    # Save CNN training history
    with open('models/' + NETWORK_NAME + '_history_' + 'fold_' + str(i + 1) + '.pkl', 'wb') as f:
        pickle.dump(history.history, f)  # Save the history object

    end = time.time()
    training_time = np.round((end - start)/60, 2)
    print('CNN training completed.')
    print('CNN training time: ' + str(training_time) + 'min(s)' + '.\n')

# Load accuracy_list, val_accuracy_list, loss_list, val_loss_list, average_final_acc, final_std_accuracy
with open('training_acc_loss_lists/' + NETWORK_NAME + '_accuracy_list' + '.npy', 'rb') as f:
    accuracy_list = np.load(f)
with open('training_acc_loss_lists/' + NETWORK_NAME + '_val_accuracy_list' + '.npy', 'rb') as f:
    val_accuracy_list = np.load(f)
with open('training_acc_loss_lists/' + NETWORK_NAME + '_loss_list' + '.npy', 'rb') as f:
    loss_list = np.load(f)
with open('training_acc_loss_lists/' + NETWORK_NAME + '_val_loss_list' + '.npy', 'rb') as f:
    val_loss_list = np.load(f)
with open('final_acc_std/' + NETWORK_NAME + '_final_avg_accuracy' + '.npy', 'rb') as f:
    average_final_acc = np.load(f)
with open('final_acc_std/' + NETWORK_NAME + '_final_std_accuracy' + '.npy', 'rb') as f:
    std_dev_final_acc = np.load(f)

# Plot graphs and classification report
results_dir = 'results/'
plotter = Plotter(NETWORK_NAME, results_dir)
# Create line graph for average acc and val acc
plotter.plot_average_acc_line_graph(accuracy_list, val_accuracy_list)
# Create line graph for average validation loss and loss
plotter.plot_average_loss_line_graph(loss_list, val_loss_list)

# Print final accuracy
print('Final Accuracy: ' + str(average_final_acc) + '±' + str(std_dev_final_acc))

# Compare both models
if COMPARE_NETWORKS:
    network1 = 'T-CNN'
    network2 = 'LeNet5'

    with open('final_acc_std/' + network1 + '_final_avg_accuracy' + '.npy', 'rb') as f:
        average_final_acc_network1 = np.load(f)
    with open('final_acc_std/' + network2 + '_final_avg_accuracy' + '.npy', 'rb') as f:
        average_final_acc_network2 = np.load(f)

    with open('final_acc_std/' + network1 + '_final_std_accuracy' + '.npy', 'rb') as f:
        std_dev_final_acc_network1 = np.load(f)
    with open('final_acc_std/' + network2 + '_final_std_accuracy' + '.npy', 'rb') as f:
        std_dev_final_acc_network2 = np.load(f)

    plotter.plot_bar_graph(network1, average_final_acc_network1, std_dev_final_acc_network1,
                           network2, average_final_acc_network2, std_dev_final_acc_network2)





