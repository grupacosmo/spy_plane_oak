from Scripts.data_preprocessing import Preprocessing
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


class Model:
    def __init__(self, prepared_data:Preprocessing):
        self.prepared_data = prepared_data
        self.size = prepared_data.size
        self.model = Sequential()

    def build_cnn_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(self.size, self.size, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(self.size, self.size, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

    # TODO: wrzucić obrobione dane
    def fit_model(self):
        self.model.fit()

    def validate_model(self):
        self.model.evaluate()

    def save_model(self):
        self.model.save("model.h5")
