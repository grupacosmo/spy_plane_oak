import os
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import shutil


class Data():
    def __init__(self):
        self.paths = {
            'train': 'data/train',
            'validation': 'data/validation',
            'test': 'data/test'
        }
        self.data = {'train': [], 'validation': [], 'test': []}
        self.prepare_and_load_data()

    def prepare_and_load_data(self):
        # if not os.listdir('./data/pickle'):
        #     for set_name in self.paths:
        #         if self.paths[set_name]:
        #             self.data[set_name] = self.prepare_data(self.paths[set_name])
        #             self.save_data(self.data[set_name], set_name)
        #             self.data[set_name] = self.load_data(set_name)
        # else:
        #     for set_name in self.paths:
        #         self.data[set_name] = self.load_data(set_name)
        if any(not os.listdir(path) for path in self.paths.values()):
            print("weszlo")
            all_data = self.prepare_dataset()
            self.data['train'], self.data['validation'], self.data['test'] = all_data
            for set_name in self.paths:
                self.save_data(self.data[set_name], set_name)
            self.copy_files_to_folders()
        else:
            print("nie weszło")
            for set_name in self.paths:
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

    def prepare_dataset(self):
        base_path = 'data/dataset'
        categories = {'cars': 1, 'non_cars': 0}
        file_paths = {'images': [], 'labels': []}
        data = {'images': [], 'labels': []}

        for category, label in categories.items():
            category_path = os.path.join(base_path, category).replace("\\", "/")
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file).replace("\\", "/")
                image = cv2.imread(file_path)
                image = cv2.resize(image, (100, 100))
                image = np.array(image)
                data['images'].append(image)
                data['labels'].append(label)
                file_paths['images'].append(file_path)
                file_paths['labels'].append(label)

        images = np.array(data['images'])
        labels = np.array(data['labels'])

        train_images, temp_images, train_labels, temp_labels, train_indices, temp_indices = train_test_split(
            images, labels, range(len(labels)), test_size=0.2, stratify=labels, random_state=42, shuffle=True)
        test_images, val_images, test_labels, val_labels, test_indices, val_indices = train_test_split(
            temp_images, temp_labels, temp_indices, test_size=0.5, stratify=temp_labels, random_state=42, shuffle=True)

        self.indices = {
            'train': train_indices,
            'validation': val_indices,
            'test': test_indices
        }
        self.file_paths = file_paths

        train_data = {'images': train_images, 'labels': train_labels}
        val_data = {'images': val_images, 'labels': val_labels}
        test_data = {'images': test_images, 'labels': test_labels}

        return train_data, val_data, test_data

    def copy_files_to_folders(self):
        for set_name in ['train', 'validation', 'test']:
            target_folder = self.paths[set_name]
            indices = self.indices[set_name]
            for idx in indices:
                label = 'cars' if self.file_paths['labels'][idx] == 1 else 'non_cars'
                src_file_path = self.file_paths['images'][idx]
                dest_folder = os.path.join(target_folder, label)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                shutil.copy(src_file_path, dest_folder)

    def save_data(self, data, file_name):
        file_path = f'data/pickle/{file_name}.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def load_data(self, filename):
        file_path = f'data/pickle/{filename}.pkl'
        with open(file_path, 'rb') as file:
            return pickle.load(file)

if __name__ == '__main__':
    data = Data()