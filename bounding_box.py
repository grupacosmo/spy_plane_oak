
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import re
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
INIT_LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 16


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == "__main__":
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


    # for image in os.listdir('dron'):
    #     img = cv2.imread(f"dron/{image}")
    #     print(img.shape)
    img = cv2.imread(f'Old_approach/dron/6.PNG')
    img = cv2.resize(img, (500, 250))
    w2, h2 = img.shape[:2]
    target_coor = []
    target_coor_temp = []
    data_image = []
    pic = {}
    car_in_pic = {}
    # x = tqdm()
    dir_list = os.listdir('Old_approach/dron/default')
    dir_txt = []
    for dir in dir_list:
        if dir.endswith(".txt"):
            dir_txt.append(dir)
    image_list = os.listdir('Old_approach/dron')
    for dir, image in zip(dir_txt, image_list):
        img = cv2.imread(f'Old_approach/dron/{image}')
        w1, h1 = img.shape[:2]
        if dir.endswith(".txt") and image.endswith(".PNG"):
            with open(f"Old_approach/dron/default/{dir}", "r") as file:
                lines = file.readlines()
                for line in tqdm(lines):
                    var1, var2, var3, var4 = line.split()
                    target_coor_temp.append([float(var1)/h1, float(var2)/w1, float(var3)/h1, float(var4)/w1])
                    imgg = load_img(f'Old_approach/dron/{image}', target_size=(500, 250))
                    imgg = img_to_array(imgg)
                    data_image.append(imgg)
            num = re.findall(r'\d+', dir)

            pic[f"{num[0]}"] = target_coor_temp
            target_coor = target_coor + target_coor_temp
            target_coor_temp=[]

    # for photo in pic.items():
    #     im, cars = photo
    #     imgggg = cv2.imread(f'dron/{im}.PNG')
    #     imgggg = cv2.resize(imgggg, (1000, 500))
    #     for car in cars:
    #
    #         xy, hw = (int(car[0]*h2),int(car[1]*w2)), (int(car[2]*h2), int(car[3]*w2))
    #     # xy, hw = (int(float(car[0])),int(float(car[1]))), (int(float(car[2])),int(float(car[3])))
    #         cv2.rectangle(imgggg, xy, hw, color = (0,255,0), thickness=1)
    #     cv2.imshow('image', imgggg)
    #     cv2.waitKey(0)

    a = list(pic.values())
    data_image = np.array(data_image, dtype="float32")/255.0
    target_coor = np.array(target_coor, dtype="float32")
    # target_coor = np.array(a, dtype="float32")

    data_image_train, data_image_test, target_coor_train, target_coor_test = train_test_split(data_image, target_coor, test_size=0.10,
                             random_state=42)

    ...
    # load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=(500, 250, 3)))
    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)

    opt = Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    H = model.fit(
        data_image_train, target_coor_train,
        validation_data=(data_image_test, target_coor_test),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1)

    print("[INFO] saving object detector model...")
    model.save("model_500_250.keras")
    N = NUM_EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("wykres500_250.png")





