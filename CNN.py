import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras import backend as K
from ImageHandler import Imagehandler
from utility import Pyramid,SlidingWindow
from matplotlib import pyplot as plt
import yaml
import glob
import os
import random
import time
import cv2 as cv
import numpy as np

def train_CNN():
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    IOPlaces = cfg['Main']
    Input = IOPlaces['Input']
    directorypathpos = Input['positive']
    os.chdir(directorypathpos)
    filesTypes = cfg['FileType']
    images = []
    for filetype in filesTypes:
        images.extend(glob.glob("*." + filetype))
    DataX = []
    DataY = []
    print('ready to load pictures')
    time_start = time.time()
    paths = [directorypathpos + image for image in images]

    for i in range(len(paths)):
        obj = Imagehandler(paths[i])
        DataX.append(obj.Image)
        DataY.append(1)

    directorypathneg = Input['Negative']
    os.chdir(directorypathneg)
    images = []
    for filetype in filesTypes:
        images.extend(glob.glob("*." + filetype))
    paths = [directorypathneg + image for image in images]
    for i in range(len(paths)):
        Image = cv.imread(paths[i], cv.IMREAD_UNCHANGED)
        for j in range(10):
            rand = random.randint(0, 50)
            img = Image[rand:rand + 128, rand:rand + 64]
            obj = Imagehandler(paths[i], img)
            DataX.append(obj.Image)
            DataY.append(0)

    time_end = time.time()
    print('use %d seconds'%(time_end - time_start))
    batch_size = 32
    num_classes = 2
    epochs = 12
    DataX = np.array(DataX)
    DataY = np.array(DataY)
# input image dimensions
    img_rows, img_cols = 128, 64
    x_train, x_test, y_train, y_test = train_test_split(DataX, DataY, test_size=0.25, random_state=30)


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save('D:/thunderdownload/mnist__data/hog/cnn_model.h5')

    IOPlaces = cfg['Main']
    filesTypes = cfg['FileType']
    output = IOPlaces['Output']
    os.chdir(output)
    images = []
    for filetype in filesTypes:
        images.extend(glob.glob("*." + filetype))
    paths = [output + image for image in images]
    for i in range(len(paths)):
        # for i in range(5):
        Image = cv.imread(paths[i], cv.IMREAD_UNCHANGED)
        imageHeight, imageWidth = Image.shape[:2]
        imageHeight = int(imageHeight / 128) * 128
        imageWidth = int(imageWidth / 64) * 64
        Image = cv.resize(Image, (imageWidth, imageHeight), interpolation=cv.INTER_CUBIC)
        print(Image.shape)
        scale_x = []
        scale_y = []
        power = []
        for (scaledImage, times) in Pyramid(Image, 2, (128, 64)):
            for (x, y, window) in SlidingWindow(scaledImage, (28, 28), (64, 128)):
                # print(window.shape[:2])
                # plt.imshow(window)
                # plt.show()
                if window.shape[:2] != (128, 64):
                    continue
                oi = Imagehandler(paths[i], window)
                if K.image_data_format() == 'channels_first':
                    window = np.reshape(oi.Image, (1, 3, img_rows, img_cols))
                else:
                    window = np.reshape(oi.Image, (1, img_rows, img_cols, 3))
                #proba = model.predict_classes(window, verbose=0)
                proba = model.predict_classes(window, verbose=0)
                clone = scaledImage.copy()
                cv.rectangle(clone, (x, y), (x + 64, y + 128), (0, 255, 0), 2)
                if proba == 1 and times > 1.5: #暂时这么处理
                    scale_x.append(x)
                    scale_y.append(y)
                    power.append(times)
                cv.imshow("Window", clone)
                cv.waitKey(1)
        Image = cv.cvtColor(Image, cv.COLOR_BGR2RGB)
        for point in range(len(scale_x)):
            cv.rectangle(Image, (scale_x[point] * power[point], scale_y[point] * power[point]),
                         (scale_x[point] * power[point] + 64 * power[point],
                          scale_y[point] * power[point] + 128 * power[point]),
                         (255, 0, 0), 2)

        plt.imshow(Image)
        plt.show()