import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import itertools
import pywt.data
import pywt
import cv2
import numpy as np
import time
import errno
import os
# from google.colab import drive
from skimage.color import rgb2lab, lab2rgb
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
import copy
import joblib
from sklearn.svm import SVR


"""# Utilities"""


def saveFrame(output_loc, frames=[], fileName="", isSave=False, isRGB2BGR=False):
    frame_number = 0
    fNameList = []
    for f in frames:
        fName = "{}/{} - {}.jpg".format(output_loc, fileName, (frame_number+1))
        fNameList.append(fName)
        if isSave:
            f_copy = copy.deepcopy(f)
            if isRGB2BGR:
                f_copy = cv2.cvtColor(f_copy, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fName, f_copy)
        frame_number += 1
    return fNameList


def saveOneFrame(output_loc, frame, fileName="", isSave=False, isRGB2BGR=False):
    fName = "{}/{}.jpg".format(output_loc, fileName)
    if isSave:
        f_copy = copy.deepcopy(frame)
        if isRGB2BGR:
            f_copy = cv2.cvtColor(f_copy, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fName, f_copy)


def showFrame(Xsub_rgb, title="original RGB", channel=["R", "G", "B"], Nsample=1, isShow=False):
    if isShow:
        count = 1
        fig = plt.figure(figsize=(12, 3*Nsample))
        # This section plot the original Xsub_rgb
        ax = fig.add_subplot(Nsample, 4, count)
        Xsub_rgb_copy = np.array(copy.deepcopy(Xsub_rgb))

        ax.imshow(Xsub_rgb_copy/255.0)
        ax.axis("off")
        ax.set_title(title)
        count += 1

        for i, lab in enumerate(channel):
            crgb = np.zeros(Xsub_rgb_copy.shape)
            crgb[:, :, i] = Xsub_rgb_copy[:, :, 0]
            ax = fig.add_subplot(Nsample, 4, count)
            ax.imshow(crgb/255.0)
            ax.axis("off")
            ax.set_title(lab)
            count += 1

        plt.show()


def saveDataframeAsFile(df, fileName='out', trial='trial-undefined', isZip=False):
    out = '../content/drive/My Drive/A TA/generated_df/'+trial+'/'
    try:
        os.makedirs(out)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        # time.sleep might help here
        pass
    if isZip:
        compression_opts = dict(method='zip', archive_name=(fileName+'.csv'))
        df.to_csv((out+fileName+'.zip'), index=False,
                  compression=compression_opts)
    else:
        df.to_csv((out+fileName+'.csv'), index=False)
