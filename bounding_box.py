
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
NUM_EPOCHS = 25
BATCH_SIZE = 32

if __name__ == "__main__":
    # for image in os.listdir('dron'):
    #     img = cv2.imread(f"dron/{image}")
    #     print(img.shape)
    img = cv2.imread("dron/5.PNG")
    w1, h1 = img.shape[:2]
    img = cv2.resize(img,(1000, 500))
    w2, h2 = img.shape[:2]
    target_coor = []
    target_coor_temp = []
    data_image = []
    pic = {}
    car_in_pic = {}
    # x = tqdm()
    dir_list = os.listdir('dron/default')
    dir_txt = []
    for dir in dir_list:
        if dir.endswith(".txt"):
            dir_txt.append(dir)
    image_list = os.listdir('dron')
    for dir, image in zip(dir_txt, image_list):
        if dir.endswith(".txt") and image.endswith(".PNG"):
            with open(f"dron/default/{dir}", "r") as file:
                lines = file.readlines()
                for line in tqdm(lines):
                    var1, var2, var3, var4 = line.split()
                    target_coor_temp.append([float(var1)/h1, float(var2)/w1, float(var3)/h1, float(var4)/w1])
                    # imgg = cv2.imread(f'dron/{image}')
                    imgg = load_img(f'dron/{image}', target_size=(1000, 500))
                    imgg = img_to_array(imgg)
                    data_image.append(imgg)
            num = re.findall(r'\d+', dir)

            pic[f"{num[0]}"] = target_coor_temp
            target_coor = target_coor + target_coor_temp
            target_coor_temp=[]
    for car in pic["5"]:
        xy, hw = (int(car[0]*h2),int(car[1]*w2)), (int(car[2]*h2), int(car[3]*w2))
        # xy, hw = (int(float(car[0])),int(float(car[1]))), (int(float(car[2])),int(float(car[3])))
        cv2.rectangle(img, xy, hw, color = (0,255,0), thickness=1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    #
    # data_image = np.array(data_image, dtype="float32")/255.0
    # target_coor = np.array(target_coor, dtype="float32")
    #
    # data_image_train, data_image_test, target_coor_train, target_coor_test = train_test_split(data_image, target_coor, test_size=0.10,
    #                          random_state=42)
    #
    # ...
    # # load the VGG16 network, ensuring the head FC layers are left off
    # vgg = VGG16(weights="imagenet", include_top=False,
    #             input_tensor=Input(shape=(1000, 500, 3)))
    # # freeze all VGG layers so they will *not* be updated during the
    # # training process
    # vgg.trainable = False
    # # flatten the max-pooling output of VGG
    # flatten = vgg.output
    # flatten = Flatten()(flatten)
    # # construct a fully-connected layer header to output the predicted
    # # bounding box coordinates
    # bboxHead = Dense(128, activation="relu")(flatten)
    # bboxHead = Dense(64, activation="relu")(bboxHead)
    # bboxHead = Dense(32, activation="relu")(bboxHead)
    # bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # # construct the model we will fine-tune for bounding box regression
    # model = Model(inputs=vgg.input, outputs=bboxHead)
    #
    # opt = Adam(lr=INIT_LR)
    # model.compile(loss="mse", optimizer=opt)
    # print(model.summary())
    # # train the network for bounding box regression
    # print("[INFO] training bounding box regressor...")
    # H = model.fit(
    #     data_image_train, target_coor_train,
    #     validation_data=(data_image_test, target_coor_test),
    #     batch_size=BATCH_SIZE,
    #     epochs=NUM_EPOCHS,
    #     verbose=1)
    #
    # print("[INFO] saving object detector model...")
    # model.save("model_RCNN.keras")
    # N = NUM_EPOCHS
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    # plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    # plt.title("Bounding Box Regression Loss on Training Set")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss")
    # plt.legend(loc="lower left")
    # plt.savefig("wykres.png")





