from Old_approach.Scripts.gather_data import Data
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

    def normalize(self):
        for img_set in self.data.data:
            self.data.data[img_set]['images'] = self.data.data[img_set]['images']/255
    def resize(self):
        for img_set in self.data.data:
            a, b, c, d = (self.data.data[img_set]['images'].shape)
            shap = (a,) + self.size + (3,)
            resized_imgs = np.ndarray(shape=shap)
            for index, img in enumerate(self.data.data[img_set]['images']):
                img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
                resized_imgs[index] = img
            self.data.data[img_set]['images'] = resized_imgs
            # a, b, c, d = (self.data.data[img_set]['images'].shape)
            # shap = (a,) + self.size
            # self.data.data[img_set]['images'] = cv2.resize(self.data.data[img_set]['images'], self.size)

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
