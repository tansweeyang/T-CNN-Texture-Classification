import os
import cv2
import numpy as np
from keras.utils import np_utils


class ImageLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.categories = []

    def read_cv_content(self, train_file_name, val_file_name, test_file_name):
        # Load split file content into an array
        # Get predefined split file path according to split_name

        train_path = self.dataset_path + 'labels/' + train_file_name + '.txt'
        val_path = self.dataset_path + 'labels/' + val_file_name + '.txt'
        test_path = self.dataset_path + 'labels/' + test_file_name + '.txt'

        with open(train_path, 'r') as f:
            train_f_content = f.read().splitlines()
        with open(val_path, 'r') as f:
            val_f_content = f.read().splitlines()
        with open(test_path, 'r') as f:
            test_f_content = f.read().splitlines()

        return train_f_content, val_f_content, test_f_content

    def load_predefined_split_images(self, splits_file_content, img_size):
        images = []
        labels = []

        # Load images matched with split file content
        # Get category path
        categories_path = self.dataset_path + 'images'
        # Get list of categories
        self.categories = os.listdir(categories_path)

        # For every category
        for i, category in enumerate(self.categories):
            category_path = os.path.join(self.dataset_path, 'images', category)  # Get category path
            image_files = os.listdir(category_path)  # Get image files

            # for every image file
            for image_file in image_files:
                # Check if the image file name matches with the split file content
                split_image_path = category + '/' + image_file  # Get image path (in split file content format)
                if split_image_path in splits_file_content:
                    image_path = os.path.join(self.dataset_path, 'images', category, image_file)  # Get image path
                    image = cv2.imread(image_path, 0)  # Open image in grayscale mode from path
                    image = cv2.resize(image, (img_size, img_size))

                    images.append(image)  # Append image to array
                    labels.append(i)

        images = np.array(images)
        labels = np.array(labels)

        labels = np_utils.to_categorical(labels, num_classes=len(self.categories))  # One-hot encode labels
        return images, labels

    @staticmethod
    def preprocess_images(images):
        preprocessed_images = []

        for image in images:
            equalized_image = cv2.equalizeHist(image)  # Equalize image
            preprocessed_images.append(equalized_image)

        preprocessed_images = np.stack(preprocessed_images, axis=0)  # Stack preprocessed images along a new axis
        return preprocessed_images

    @staticmethod
    def normalize_images(images):
        normalized_images = []

        for image in images:
            normalized_image = (image - np.mean(image)) / np.std(image)
            normalized_images.append(normalized_image)

        normalized_images = np.stack(normalized_images, axis=0)  # Stack normalized images along a new axis
        return normalized_images
