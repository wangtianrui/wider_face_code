import scipy.io
import cv2


data=scipy.io.loadmat("G:/baiduyun/faceresource/wider/wider_face_split/wider_face_train.mat")
#data['file_list']里面的file_list就是标签的名字。由于data是字典格式。所以可以用data.keys()去查询一下它所有标签的名字。
FileName=data['file_list']
FileValue=data['face_bbx_list']
FileOcc=data['occlusion_label_list']
#FileName[n][x][k][y]中，n是第n个图像集（比方这里有共61有类），x是只能取0，因为file_list只有1列。k是第k个图片。y也只能取0.
print(FileName[0][0][0][0])
num = 0
for k in range(61):
    for x in FileName[k][0]:
        num = num + 1

        print (x[0][0],num)

for x in FileName[0][0]:
    print (x[0][0])