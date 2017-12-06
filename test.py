import scipy.io
import os
import cv2
data = scipy.io.loadmat("G:/baiduyun/faceresource/wider/wider_face_split/wider_face_train.mat")
# data['file_list']里面的file_list就是标签的名字。由于data是字典格式。所以可以用data.keys()去查询一下它所有标签的名字。
FileName = data['file_list']
FileValue = data['face_bbx_list']
FileOcc = data['occlusion_label_list']
# FileName[n][x][k][y]中，n是第n个图像集（比方这里有共61有类），x是只能取0，因为file_list只有1列。k是第k个图片。y也只能取0.
#print(FileName[10][0][0][0])
num = 0
noicecount = 0
count = 0
TIMES = 0
index = 0
picNum = 0
RandCor = []
FaceCor = []

for k in range(61):
    for x in FileName[k][0][0][0]:
        print(x)

'''
for x in FileName:
    print (x)
     G:/SavedPictures/timg.jpg
'''

print(os.path.exists('G:/SavedPictures/timg1.jpg'))
image = cv2.imread("G:/SavedPictures/timg.jpg")
#if(image == None):
    #print("test")
#print(image)

def file_name(step):
    #print(step)
    file_dir = 'G:/baiduyun/faceresource/wider/WIDER_train/images/'
    if (step >= 10):
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # 当前目录路径
            for f in dirs:
                #print(f[1])
                if(f[1]!='-'):
                   # print("step:",step,f[:2])
                    if (step == int(f[:2])):
                       # print("test1:", f[:2])

                        file_dir = file_dir + f + "/"
                        #print("testaddfile:", file_dir)
                '''
                if (f[1] == '-'):
                    print(f[0])
                else:
                    print(f[:2])  # 当前路径下所有子目录
                    # print(files)  # 当前路径下所有非目录子文件
                    # filelist = os.walk(file_dir)
                    # print(filelist[])
                '''
    else:
        #print("小于10")
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # 当前目录路径
            for f in dirs:
                print("printftest:",f[:2])
                if (step == int(f[0]) and f[1]== '-'):
                    #print("test2:", f[0])
                    file_dir = file_dir + f + "/";
                '''
                if (f[1] == '-'):
                    print(f[0])
                else:
                    print(f[:2])  # 当前路径下所有子目录
                    # print(files)  # 当前路径下所有非目录子文件
                    # filelist = os.walk(file_dir)
                    # print(filelist[])
                '''
    return file_dir


def FindAddress(step):
    # print("test3:",step)
    string = file_name(step)
    return string


for step in range(10,61):
    log = open("log.txt", 'w')
    # i遍历FileName[step][0],j遍历FileValue[step][0]，p遍历FileOcc[step][0]
    for (i, j, p) in zip(FileName[step][0], FileValue[step][0], FileOcc[step][0]):
        # print("test4", step)
        #step = 12
        newstring = FindAddress(step)
        #print("THE k is", step)
        if (newstring == 0):
            continue
        newstring = newstring + i[0][0] + '.jpg'
        num = num + 1
        #print(newstring, num)
        if (newstring == 0):
            continue
        newstring = newstring + i[0][0] + '.jpg'
        num = num + 1
        #print(newstring)
        '''
        img = cv2.imread(newstring)
        if (img == None):
            continue
        FaceCor = []
        

        for k in range(j[0].shape[0]):
            #print("test:", j[0][k])
            if int(j[0][k][2]) <= 50 and int(j[0][k][3]) <= 50 or p[0][k] != 0:
                continue
         
            crop = img[int(j[0][k][1]):int(j[0][k][1]) + int(j[0][k][3]),
                   int(j[0][k][0]):int(j[0][k][0]) + int(j[0][k][2])]
            count += 1
            filename = "F:/Traindata/facedata/activedata/" + str(count) + ".jpg"
            cv2.imwrite(filename, crop)
            print(count)
            FaceCor = FaceCor + [j[0][k]]
        print(FaceCor)
        if (FaceCor != []):
            picNum = 0
            RandCor = RandomClipPhoto(FaceCor, img.shape[1], img.shape[0])
            for w in range(len(RandCor)):
                if (picNum == 4):
                    break
                picNum += 1
                crop2 = img[RandCor[w][1]:RandCor[w][1] + RandCor[w][3], RandCor[w][0]:RandCor[w][0] + RandCor[w][2]]
                filename2 = "F:/Traindata/facedata/negativedata/" + str(noicecount) + ".jpg"
                noicecount += 1
                cv2.imwrite(filename2, crop2)
        TIMES += 1  # the processing image num
    log.write(i[0][0] + "  finish")
    log.close()
    '''