import cv2

imagePath = "G:/baiduyun/faceresource/wider/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg"

def cutImage(image , top , left , width , height):
    image = image[left:left+height,top:top+width]
    return image
image = cv2.imread(imagePath)
#image = image[330:479,449:571]
image = cutImage(image,449,330,122,149)
cv2.imshow("test",image)
cv2.waitKey(0)