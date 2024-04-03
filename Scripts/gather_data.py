import os
import pickle
import numpy as np
import cv2


#TODO
# Klasa z metodami do:
# - Niech przy tworzeniu obiektu od razu tworzył 3 zmienne z danymi do trenowania, walidacji i testowania
# (więc  trzeba dodać 3 plik), najlepiej wrzuć wszystkie dane do 1 pliku i potem to podziel na 3
# a w klasie zrób może liste list data[train[...], validation[...], test[...]]

class Data():
    def __init__(self):
        self.paths = {
            'train': './data/train',
            'validation': './data/validation',
            #'test': '../data'
        }
        self.data = {'train': [], 'validation': []} # 'test': []
        if not os.listdir('./data/pickle'):
            self.prepare_and_load_data()

    def prepare_and_load_data(self):
        for set_name in self.paths:
            if self.paths[set_name]:
                self.data[set_name] = self.prepare_data(self.paths[set_name])
                self.save_data(self.data[set_name], set_name)
                self.data[set_name] = self.load_data(set_name)

    def prepare_data(self, path):
        data = {'images': [], 'labels': []}
        categories = {'cars': 1, 'non_cars': 0}
        for category, label in categories.items():
            categories_path = os.path.join(path, category).replace("\\","/")
            for file in os.listdir(categories_path):
                full_path = os.path.join(categories_path, file).replace("\\","/")
                image = cv2.imread(full_path)
                image = np.array(image)
                data['images'].append(image)
                data['labels'].append(label)
        return data

    def save_data(self, data, file_name):
        file_path = f'./data/pickle/{file_name}.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def load_data(self, filename):
        file_path = f'./data/pickle/{filename}.pkl'
        with open(file_path, 'rb') as file:
            return pickle.load(file)


# train_path = '../data/train'
# validation_path = '../data/validation'
# #test_path = '../data/test'
#
#
# dataset = Data(train_path, validation_path) #test_path
