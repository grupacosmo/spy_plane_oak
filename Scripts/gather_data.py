import os
import pickle
import numpy as np
from PIL import Image


#TODO
# Klasa z metodami do:
# - Ładowania danych z pickle
# - Tworzenia pickle z dostępnych zdjeć
# -


class Data():
    def __init__(self):
        pass





# data paths
train_path = '../data/train'
validation_path = '../data/validation'



# prepare labels for data
def prepare_data(path):
    data = {'images': [], 'labels': []}
    categories = {'cars': 1, 'non_cars': 0}
    for category, label in categories.items():
        categories_path = os.path.join(path, category)
        for file in os.listdir(categories_path):
            full_path = os.path.join(categories_path, file)
            image = Image.open(full_path)
            image = np.array(image)
            data['images'].append(image)
            data['labels'].append(label)
    return data


# save prepared labels in pickle
def save_data(data, file):
    file = '../data/' + file + '.pkl'
    with open(file, 'wb') as f:
        pickle.dump(data, f)


# load prepared data from pickle
def load_data(file):
    file = '../data/' + file + '.pkl'
    with open(file, 'rb') as f:
        return pickle.load(f)


train_data = prepare_data(train_path)
save_data(train_data, 'train_data')

validation_data = prepare_data(validation_path)
save_data(validation_data, 'validation_data')

