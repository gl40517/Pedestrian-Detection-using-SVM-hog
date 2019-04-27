import cv2 as cv

def Pyramid(img,scale,minsize):
    origin_height, origin_width = img.shape[:2]
    "this function is used to create the guassian pyramid of the images"
    yield img, 1

    imageHeight,imageWidth=img.shape[:2]
    while imageHeight >minsize[0]  and imageWidth>minsize[1]:
        imageHeight=int(imageHeight/scale)
        imageWidth=int(imageWidth/scale)
        img=cv.resize(img,(imageWidth,imageHeight), interpolation = cv.INTER_CUBIC)
        yield img, int(origin_height/imageHeight)
        
def SlidingWindow(image, stepSize, windowSize):
    for x in range(0, image.shape[1], stepSize[0]):
        for y in range(0, image.shape[0], stepSize[1]):
            yield (x, y, image[y:y + windowSize[1],x:x + windowSize[0] ])
