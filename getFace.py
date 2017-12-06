import scipy.io
import os
import cv2

count = 0
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


def file_name(temp):
    # print(temp)
    file_dir = 'G:/baiduyun/faceresource/wider/WIDER_train/images/'
    if (temp >= 10):
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # 当前目录路径
            for f in dirs:
                if (f[1] != '-'):
                    if (temp == int(f[:2])):
                        #print("test1:", f[:2])
                        file_dir = file_dir + f + "/";

    else:
        # print("小于10")
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # 当前目录路径
            for f in dirs:
                # print("printftest:",f[0])
                if (temp == int(f[0]) and f[1] == '-'):
                    # print("test2:", f[0])
                    file_dir = file_dir + f + "/";

    if(file_dir == 'G:/baiduyun/faceresource/wider/WIDER_train/images/'):
        file_dir = 'error'
    return file_dir


def FindAddress(step):
    if (step < 2):
        temp = step
    elif (step >= 2 and step <= 11):
        temp = 10 + step - 2
    # print("test3:",step)
    elif (step == 12):
        temp = 2
    elif (step > 12 and step <= 22):
        temp = 20 + step - 13
    elif (step == 23):
        temp = 3
    elif (step > 23 and step <= 33):
        temp = 30 + step - 24
    elif (step == 34):
        temp = 4
    elif (step > 34 and step <= 44):
        temp = 40 + step - 35
    elif (step == 45):
        temp = 5
    elif (step > 45 and step <= 55):
        temp = 50 + step - 46
    elif (step == 56):
        temp = 6
    elif (step > 56 and step <= 66):
        temp = 60 + step - 57
    string = file_name(temp)
    #print(step)
    return string


for step in range(61):
    print(step)
    #log = open("log.txt", 'w')
    # i遍历FileName[step][0],j遍历FileValue[step][0]，p遍历FileOcc[step][0]
    for (i, j, p) in zip(FileName[step][0], FileValue[step][0], FileOcc[step][0]):
        # print("test4", step)
        #
        #print("THE k is", step)
        newstring = FindAddress(step)
        #print("findAddok")
        if (newstring == 'error'):
            print("string error continue")
            continue
        #print("readok")
        newstring = newstring + i[0][0] + '.jpg'
        num = num + 1
        print(newstring,num)
        if (os.path.exists(newstring) == False):
            print("path error")
            continue
        img = cv2.imread(newstring)
        print("readok")
        #FaceCor = []
        for k in range(j[0].shape[0]):  # 人头数
            if int(j[0][k][2]) <= 50 and int(j[0][k][3]) <= 50 or p[0][k] != 0:
                continue
            crop = img[int(j[0][k][1]):int(j[0][k][1]) + int(j[0][k][3]),
                   int(j[0][k][0]):int(j[0][k][0]) + int(j[0][k][2])]
            count += 1
            print("write")
            filename = "F:/Traindata/facedata/activedata/" + str(count) + ".jpg"
            cv2.imwrite(filename, crop)
        print(step)
        print("writeok")
