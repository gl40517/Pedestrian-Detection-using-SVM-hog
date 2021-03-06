from ImageHandler import Imagehandler
from svmFile import svmClass
from utility import Pyramid,SlidingWindow
import yaml
import glob
import os
import random
import cv2 as cv
from matplotlib import pyplot as plt
import time
import CNN


def App():
    print('use SVM(0) or CNN(1):')
    flag = eval(input())
    if not flag:
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        IOPlaces = cfg['Main']
        Input = IOPlaces['Input']
        output = IOPlaces['Output']
        directorypathpos = Input['positive']
        os.chdir(directorypathpos)
        filesTypes = cfg['FileType']
        images=[]
        for filetype in filesTypes:
            images.extend(glob.glob("*."+filetype))
        DataX = []
        DataY = []
        print('ready to load pictures')
        time_start = time.time()
        paths=[directorypathpos + image for image in images]
###adding and Computing HOG Vector
    
        for i in range(len(paths)):
            obj = Imagehandler(paths[i])
            HogVector = obj.ImagesToTiles(16,16)
            DataX.append(HogVector)
            DataY.append(1)
        
        directorypathneg = Input['Negative']
        os.chdir(directorypathneg)
        images = []
        for filetype in filesTypes:
            images.extend(glob.glob("*."+filetype))
        paths = [directorypathneg + image for image in images]
        for i in range(len(paths)):
            Image = cv.imread(paths[i],cv.IMREAD_UNCHANGED)
            for j in range(10):
                rand = random.randint(0,50)
                img = Image[rand:rand+128,rand:rand+64]
                obj = Imagehandler(paths[i],img)
                HogVector = obj.ImagesToTiles(16, 16)
                DataX.append(HogVector)
                DataY.append(0)
    

    
 ###train and test   
        print('picture loading done')
        time_end = time.time()
        print('use %d seconds'%(time_end - time_start))


        print('ready to train SVM')
        svmObj = svmClass((DataX,DataY))
        svmObj.trainData()
        print('SVM train done')
####predict
        os.chdir(output)
        images = []
        for filetype in filesTypes:
            images.extend(glob.glob("*."+filetype))
        paths = [output+image for image in images]
        for i in range(len(paths)):
            Image = cv.imread(paths[i],cv.IMREAD_UNCHANGED)
            imageHeight,imageWidth = Image.shape[:2]
            imageHeight = int(imageHeight/128)*128
            imageWidth = int(imageWidth/64)*64
            Image = cv.resize(Image,(imageWidth,imageHeight), interpolation = cv.INTER_CUBIC)
            print(Image.shape)
            scale_x = []
            scale_y = []
            power = []
            for (scaledImage, times) in Pyramid(Image,2,(128,64)):
                for (x,y,window) in SlidingWindow(scaledImage,(28,28),(64,128)):
                #print(window.shape[:2])
                #plt.imshow(window)
                #plt.show()
                    if window.shape[:2] != (128,64):
                        continue
                    oi = Imagehandler(paths[i],window)
                    Hog = oi.ImagesToTiles(16,16)
                    val = svmObj.PredictData([Hog])
                    #print(val)
                    val = val[0]

                    clone = scaledImage.copy()
                    cv.rectangle(clone, (x, y), (x + 64, y + 128), (0, 255, 0), 2)

                    if val[1] > 0.9 :  # 暂时性的处理方式
                        scale_x.append(x)
                        scale_y.append(y)
                        power.append(times)
                    cv.imshow("Window", clone)
                    cv.waitKey(1)
            Image = cv.cvtColor(Image, cv.COLOR_BGR2RGB)
            for point in range(len(scale_x)):
                cv.rectangle(Image, (scale_x[point]*power[point], scale_y[point]*power[point]),
                            (scale_x[point]*power[point] + 64*power[point],
                            scale_y[point]*power[point] + 128*power[point]),
                            (255, 0, 0), 2)

            plt.imshow(Image)
            plt.show()
    else:
        CNN.train_CNN()


if __name__ == "__main__":
    
    cam = cv.VideoCapture(0)
    App()
    cam.release()
    cv.destroyAllWindows()