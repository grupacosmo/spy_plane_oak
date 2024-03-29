from Scripts.gather_data import Data
from sklearn.model_selection import train_test_split
import numpy as np
import random
import imgaug.augmenters as iaa
import cv2


class Preprocessing:
    def __init__(self, data: Data, size=(1, 1)):
        self.data = data
        self._size = size
        self.augmentation = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-30, 30)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(scale=(0.8, 1.2)),
        ])

    def augmentation(self):
        for i in range(0, 15):
            pic = random.randrange(0, 14)
        image_aug = self.augmentation('''image=pic''') #TODO: wsadzić tu obrazy i zobaczymy czy działa xd

    def resize(self):
        for img_set in self.data.data:
            for img in img_set:
                image = cv2.resize(img, self.size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_value):
        if min(new_value) <= 0:
            raise ValueError("Size must be positive!")
        elif max(new_value) > 256:
            raise ValueError("Size must be smaller than 256x256")
        self._size = tuple(int(x) for x in new_value)


    @size.getter
    def size(self):
        return self._size

    def __repr__(self):
        return repr(self.size)
