from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, \
    BatchNormalization


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.model = Sequential()

    def create_model_architecture(self, network_name, len_categories):
        if network_name == 'T-CNN':
            self.model.add(
                Conv2D(filters=96, input_shape=(227, 227, 1), kernel_size=(11, 11), strides=4, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
            self.model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation='relu'))
            self.model.add(BatchNormalization())

            self.model.add(AveragePooling2D(pool_size=(27, 27), strides=1))

            self.model.add(Flatten())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.0007)))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.0007)))
            self.model.add(Dense(units=len_categories, activation='softmax'))

        # if network_name == 'LeNet5':
        #     self.model.add(Conv2D(filters=6, input_shape=(28, 28, 1), kernel_size=(5, 5), padding='valid', activation='sigmoid'))
        #     self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
        #     self.model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
        #     self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
        #
        #     self.model.add(Flatten())
        #     self.model.add(Dense(120, activation='sigmoid'))
        #     self.model.add(Dense(84, activation='sigmoid'))
        #     self.model.add(Dense(len_categories, activation='softmax'))

        if network_name == 'LeNet5':
            self.model.add(
                Conv2D(filters=6, input_shape=(32, 32, 1), kernel_size=(5, 5), padding='valid', activation='relu'))
            self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
            self.model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
            self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

            self.model.add(Flatten())
            self.model.add(Dense(120, activation='sigmoid'))
            self.model.add(Dense(84, activation='sigmoid'))
            self.model.add(Dense(len_categories, activation='softmax'))

    def build_model(self, image_shape):
        self.model.build(image_shape)
        optimizer = Adam(learning_rate=0.001, decay=0.0005)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()

    def train_model(self, x_train, y_train, x_val, y_val, epochs, checkpoint_path):
        batch_size = 32

        checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max')

        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(x_val, y_val),
                                 callbacks=[checkpoint])

        return self.model, history
