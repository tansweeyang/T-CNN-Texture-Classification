from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, \
    BatchNormalization

class ConvolutionalNeuralNetwork:
    def __init__(self, network_name):
        self.networkName = network_name
        self.model = Sequential()

    def create_model_archiecture(self, categories):
        if self.networkName == 'T-CNN':
            self.model.add(Conv2D(filters=96, input_shape=(227, 227, 1), kernel_size=(11, 11), strides=4, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
            self.model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation='relu'))
            self.model.add(BatchNormalization())

            self.model.add(AveragePooling2D(pool_size=(27, 27), strides=1))

            self.model.add(Flatten())
            self.model.add(Dense(units=4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=4096, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=len(categories), activation='softmax'))

        if self.networkName == 'LeNet5':
            self.model.add(Conv2D(filters=6, input_shape=(227, 227, 1), kernel_size=(5, 5), strides=1, activation='tanh'))
            self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
            self.model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation='tanh'))
            self.model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
            self.model.add(Conv2D(filters=120, kernel_size=(5, 5), strides=1, activation='tanh'))

            self.model.add(Flatten())
            self.model.add(Dense(84, activation='tanh'))
            self.model.add(Dense(len(categories), activation='softmax'))

    def build_model(self, image_shape):
        self.model.build(image_shape)
        optimizer = SGD(learning_rate=0.001)
        # optimizer = Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()

    def train_model(self, X_train, y_train, X_val, y_val):
        batch_size = 20
        epochs = 100

        early_stop = EarlyStopping(monitor='val_loss', patience=5,  restore_best_weights=True)
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(X_val, y_val),
                                 callbacks=[early_stop])
        return self.model, history
