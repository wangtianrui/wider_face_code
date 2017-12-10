import scipy.io
import os
import cv2
import random

count = 17150
index = 0
picNum = 0
RandCor = []
FaceCor = []
imagePath = 'G:/baiduyun/faceresource/wider/WIDER_train/images/'
data = scipy.io.loadmat("G:/baiduyun/faceresource/wider/wider_face_split/wider_face_train.mat")
FileName = data['file_list']
FileValue = data['face_bbx_list']
FileOcc = data['occlusion_label_list']
num = 0



def FindAddress(step):
    # print(temp)
    temp = step
    file_dir = 'F:/Traindata/facedata/test/f/'
    string = file_dir + '1 (' + str(temp) + ')' + '.jpg'
    print(string)
    return string


for step in range(994):
    print(step)
    # log = open("log.txt", 'w')
    # i遍历FileName[step][0],j遍历FileValue[step][0]，p遍历FileOcc[step][0]
    '''
    for (i, j, p) in zip(FileName[step][0], FileValue[step][0], FileOcc[step][0]):
        # print("test4", step)
        #
        # print("THE k is", step)
        newstring = FindAddress(step)
        # print("findAddok")
        if (newstring == 'error'):
            print("string error continue")
            continue
        # print("readok")
        newstring = newstring + i[0][0] + '.jpg'
        num = num + 1
        print(newstring, num)
        if (os.path.exists(newstring) == False):
            print("path error")
            continue
        img = cv2.imread(newstring)
        print("readok")
        # FaceCor = []
       
        for k in range(j[0].shape[0]):  # 人头数
            if int(j[0][k][2]) <= 50 and int(j[0][k][3]) <= 50 or p[0][k] != 0:
                continue
            crop = img[int(j[0][k][1]):int(j[0][k][1]) + int(j[0][k][3]),
                   int(j[0][k][0]):int(j[0][k][0]) + int(j[0][k][2])]
            count += 1
            print("write")
            filename = "F:/Traindata/facedata/test/new/" + str(count) + ".jpg"
            cv2.imwrite(filename, crop)
     
        if (j[0].shape[0] < 5):
            for k in range(j[0].shape[0]):  # 人头数
                if int(j[0][k][2]) <= 50 and int(j[0][k][3]) <= 50 or p[0][k] != 0:
                    continue
               
                randomInt = random.randint(-200, -100)
                randomInt2 = random.randint(100, 200)
              
                if (k % 2 == 0):
                    crop = img[int(j[0][k][1] + randomInt):int(j[0][k][1] + randomInt) + int(j[0][k][3]),
                           int(j[0][k][0] + randomInt):int(j[0][k][0] + randomInt) + int(j[0][k][2])]
                else:
                    crop = img[int(j[0][k][1] + randomInt2):int(j[0][k][1] + randomInt2) + int(j[0][k][3]),
                           int(j[0][k][0] + randomInt2):int(j[0][k][0] + randomInt2) + int(j[0][k][2])]
            
                if (k % 2 == 0):
                    crop = img[int(j[0][k][1]) + 400:int(j[0][k][1]) + 400 + int(j[0][k][3]),
                           int(j[0][k][0]) + 400:int(j[0][k][0]) + 400 + int(j[0][k][2])]
                else:
                    crop = img[int(j[0][k][1]) - 400:int(j[0][k][1]) - 400 + int(j[0][k][3]),
                           int(j[0][k][0]) - 400:int(j[0][k][0]) - 400 + int(j[0][k][2])]
            '''
    imagePath = FindAddress(step)
    image = cv2.imread(imagePath)
    try:
        image1 = image[50:300, 300:550]
        count += 1
        filename = "F:/Traindata/facedata/test/new/" + str(count) + ".jpg"
        cv2.imwrite(filename, image1)

        image2 = image[300:550, 550:800]
        count += 1
        filename = "F:/Traindata/facedata/test/new/" + str(count) + ".jpg"
        cv2.imwrite(filename, image2)

        image3 = image[300:550, 300:550]
        count += 1
        filename = "F:/Traindata/facedata/test/new/" + str(count) + ".jpg"
        cv2.imwrite(filename, image3)

        image4 = image[50:300, 50:300]
        count += 1
        filename = "F:/Traindata/facedata/test/new/" + str(count) + ".jpg"
        cv2.imwrite(filename, image4)
    except TypeError:
        continue
    print("write")

print(count)