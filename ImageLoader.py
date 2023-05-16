import os
import cv2
import numpy as np
from keras.utils import np_utils
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor

class ImageLoader:
    def __init__(self, data_path, img_size, max_images_per_category):
        self.data_path = data_path
        self.img_size = img_size
        self.max_images_per_category = max_images_per_category

    def load_image(self):
        # Get the list of texture categories
        categories = os.listdir(self.data_path + "images/")

        # Initialize the arrays to store the images and labels
        images = []
        labels = []

        # Loop over the categories

        preprocessed = 0
        for i, category in enumerate(categories):
            # Get the list of images in this category
            file_list = os.listdir(self.data_path + "images/" + category)

            # Loop over the images in this category
            count = 0

            for filename in file_list:
                if count < self.max_images_per_category:
                    # Read the image and resize it to the desired size
                    img = cv2.imread(self.data_path + "images/" + category + "/" + filename)

                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))

                        # Image Preprocessing
                        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        equalized_image = cv2.equalizeHist(gray_image)

                        # Add the image and label to the arrays
                        images.append(equalized_image)
                        labels.append(i)
                    count += 1

                    preprocessed += 1
                    print('Done preprocessing image: ' + str(preprocessed) + ' Now in category: ' + str(i))
                else:
                    break

        # Convert the arrays to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Convert the labels to one-hot encoded vectors
        labels = np_utils.to_categorical(labels, num_classes=len(categories))

        return images, labels, categories
