from Old_approach.Scripts.data_preprocessing import Preprocessing

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.initializers import RandomNormal
from keras.regularizers import l2


#TODO
# 1. Make some hyperparameters tuning
class Model:
    def __init__(self, prepared_data:Preprocessing = 0):
        self.prepared_data = prepared_data
        self.size = list(prepared_data.size)
        self.model = Sequential()

    def build_cnn_model(self):
        # self.model = Sequential()
        # self.model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(self.size[0], self.size[1], 3), activation='relu'))
        # self.model.add(MaxPool2D(pool_size=(2, 2)))
        # self.model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(self.size[0], self.size[1], 3), activation='relu'))
        # self.model.add(MaxPool2D(pool_size=(2, 2)))
        # self.model.add(Flatten())
        # self.model.add(Dense(256, activation="relu"))
        # self.model.add(Dense(1, activation='softmax'))
        # self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')



        self.model.add(Conv2D(32, (5, 5), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                         kernel_regularizer=l2(0.001), input_shape=(self.size[0], self.size[1], 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (5, 5), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                         kernel_regularizer=l2(0.001), input_shape=(self.size[0], self.size[1], 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(64, (5, 5), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
        #                  kernel_regularizer=l2(0.001), input_shape=(self.size[0], self.size[1], 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None), kernel_regularizer=l2(0.001)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(self.model.summary())
    def fit_model(self):
        self.model.fit(self.prepared_data.data.data['train']['images'],
                       self.prepared_data.data.data['train']['labels'], epochs=50, batch_size=50)

    def validate_model(self):
        self.model.evaluate(self.prepared_data.data.data['validation']['images'],
                            self.prepared_data.data.data['validation']['labels'])

    def save_model(self):
        self.model.save("model.keras")
