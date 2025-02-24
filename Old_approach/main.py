import numpy as np

from Old_approach.Scripts.data_preprocessing import Preprocessing
from Old_approach.Scripts.gather_data import Data
from Old_approach.Scripts import Model
import cv2
from patchify import patchify

# TODO
#  1. Pack all into one class to reduce complexity
#  2. Test model
if __name__ == "__main__":
    data = Data()
    preprocessing = Preprocessing(data)
    preprocessing.size = (100,100)
    print(repr(preprocessing))
    preprocessing.normalize()
    preprocessing.resize()
    model = Model(preprocessing)
    model.build_cnn_model()
    # model.fit_model()
    # model.validate_model()
    # model.save_model()
    # data = Data()
    # preprocessing = Preprocessing(data)
    # model = Model(preprocessing)
    # preprocessing.size = (100, 100)
    # model.build_cnn_model()
    model.model.load_weights('model.keras')
    videofile = 'test_video.mp4'
    cap = cv2.VideoCapture(videofile)
    predicted_frames = []
    k = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.resize(frame, (640, 360))
            frame2 = frame.copy()
            predicted_pathes = []
            patches = patchify(frame, (100,100,3), step = 100)
            k = k+1
            if k == 2:
                k = 0
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i,j,:,:]
                        patch_norm_input = patch/255
                        # patch_prediction = model.model.predict(patch_norm_input)
                        frame = cv2.rectangle(frame, (j * 100, i * 100), (j * 100 + 95, i * 100 + 95), (255, 0, 0), 2, )
                        # if patch_prediction.item() > 0.5:
                        #     frame2 = cv2.rectangle(frame, (j*50, i*50), (j*50+95, i*50+95), (0, 255, 0), 2)
                        # predicted_pathes.append(patch_prediction.item())
                        # cv2.waitKey(1)
                        # alpha = 0.1
                        # frame2 = cv2.addWeighted(frame, alpha, frame2, 1 - alpha, 0)
                        cv2.imshow('frame', frame)
                        # cv2.waitKey(20)
                predicted_pathes = np.resize(predicted_pathes, (6,4))
                predicted_frames.append(predicted_pathes)

            # cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()