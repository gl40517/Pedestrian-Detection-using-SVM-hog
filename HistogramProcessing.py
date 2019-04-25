
import cv2 as cv
import numpy as np

class HistogramOperations(object):
    def __init__(self,image):
        self.Image=image
        self.ImageHeight,self.ImageWidth=image[:2]
        self.Maxangle=180
        
    def CreateHistogram(self,channel):
#        hist=cv.calcHist(self.Image,channel,mask,histSize,range)
        pass
        
    def GetGradients(self):
        gradientX=cv.Sobel(self.Image,cv.CV_64F,1,0,ksize=1)
        gradientY=cv.Sobel(self.Image,cv.CV_64F,1,0,ksize=1)
#        NormalizingConstant=np.sqrt(np.sum(gradientX*gradientX+ gradientY*gradientY))
#        gradientX=gradientX/(NormalizingConstant +1e-10)
#        gradientY=gradientY/(NormalizingConstant +1e-10)
        return gradientX,gradientY
    
    def ConvertToPolarForm(self,gradients):
        gradientX,gradientY=gradients
        mag,angle=cv.cartToPolar(gradientX,gradientY,angleInDegrees=True)
        return mag.flatten(),angle.flatten()
        #these things are not needed in the case of the PNG images and are only needed in case of three Channels
        #such as RGB Images.
        
#        tempImageTilemag=[]
#        tempImageTileangle=[]
#        for tile in mag:
#            temptile=[]
#            temptileang=[]
#            if len(tile)==3:
#                for pixel in tile:
#                    print pixel
#                    temp=np.argmax(pixel)
#                    print temp,pixel.shape
#                    temptile.append(pixel[temp])
#                    temptileang.append(pixel[temp])
#            else:
#                temptile.append(pixel)
#                temptileang.append(pixel[temp])
#            tempImageTilemag.append(temptile)
#            tempImageTileangle.append(temptileang)
#        flat_list_mag=[]
#        flat_list_ang=[]
#        for sublist in tempImageTilemag:
#            for item in sublist:
#                flat_list_mag.append(item)
#        for sublist in tempImageTileangle:
#            for item in sublist:
#                flat_list_ang.append(item)
#        tempImageTileangle=np.reshape(np.array(tempImageTilemag),(self.ImageHeight,self.ImageWidth))
#        tempImageTileangle=np.reshape(np.array(tempImageTileangle),(self.ImageHeight,self.ImageWidth))
#        return (flat_list_mag,flat_list_ang)
    
    def HistogramOfGradient(self,image,noofbins):
        gradientX,gradientY=self.GetGradients()
#        print gradientX,gradientY
        magGradientList,angleGradientList=self.ConvertToPolarForm((gradientX,gradientY))
#        print magGradientList
        bins=[0.0]*noofbins #nine bind [0,20,40,60,80,100,120,140,160]
        offset=int(self.Maxangle/noofbins) #this should be divisble for more accuracy
        #print angleGradientList
        #print magGradientList
        for i,angleGradient in enumerate(angleGradientList):
            if angleGradient >180:
                angleGradient=angleGradient-180
            leftBin=int(angleGradient/offset)
            leftBin=leftBin%noofbins
            rightBin=leftBin+1 if leftBin != noofbins-1 else 0
            rightRatio= (angleGradient-leftBin*offset)/offset
#            print rightRatio
            leftRatio=1-rightRatio
#            print leftBin,rightBin
            if leftRatio <0  or rightRatio <0:
                print (rightRatio, angleGradient,leftBin)
                print ("wait")
            bins[leftBin]=magGradientList[i]*leftRatio
            bins[rightBin]=magGradientList[i]*rightRatio
#        print( bins)
        return bins
    def ConcatAndNormalisationofHistogram(self,histogramList):
        featureVector=[]
#        print "lol",histogramList
        for i,tempHistogramList in enumerate(histogramList):
            for histogram in tempHistogramList:
                featureVector.append(histogram) 
        normSum=sum(hist*hist for hist in featureVector)
#        print "suck",featureVector
        featureVector=[feature/(normSum+1e-10) for feature in featureVector]
        #print "fuck",featureVector
        return featureVector
    @staticmethod   
    def ConcatFeatureVectors(vectors):
        imageVector=[]
        #print "done"
        for i in range(len(vectors)):
            tempvectors=vectors[i]
            [imageVector.append(vector)for vector in tempvectors]
#        print ( len(imageVector))
        return imageVector
        
    