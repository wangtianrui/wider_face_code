import cv2
import re
import linecache
import os
imagePath = "G:/baiduyun/faceresource/wider/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg"
faceLocationPath = "G:/baiduyun/faceresource/wider/wider/wider_face_split/wider_face_train/0--Parade/0_Parade_marchingband_1_849.txt"
def cutImage(image , top , left , width , height):
    image = image[left:left+height,top:top+width]
    return image
def getLocation(file):
    oneface = linecache.getline(file,0)
    print("test",oneface)
image = cv2.imread(imagePath)
#image = image[330:479,449:571]
image = cutImage(image,449,330,122,149)
getLocation(faceLocationPath)
cv2.imshow("test",image)
cv2.waitKey(0)