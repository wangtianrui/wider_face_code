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
"""
for x in FileName[0][0]:
    print (x[0][0])
"""

noicecount=0
count=0
TIMES=0
index=0
picNum=0
RandCor=[]
FaceCor=[]
data=scipy.io.loadmat("val.mat")
FileName=data['file_list']
FileValue=data['face_bbx_list']
FileOcc=data['occlusion_label_list']

for step in range(61):
    log=open("log.txt",'w')
    for (i,j,p) in zip(FileName[step][0],FileValue[step][0],FileOcc[step][0]):
        newstring=FindAddress(step)
        print( "THE k is",step)
        if(newstring==0):
            continue
        newstring=newstring+i[0][0]+'.jpg'
        print (newstring)
        img=cv2.imread(newstring)
        if(img==None):
            continue
        FaceCor=[]
        for k in range(j[0].shape[0]):
            if int(j[0][k][2])<=50 and int(j[0][k][3])<=50 or p[0][k]!=0:
		        continue
            crop=img[int(j[0][k][1]):int(j[0][k][1])+int(j[0][k][3]),int(j[0][k][0]):int(j[0][k][0])+int(j[0][k][2])]
            count+=1
            filename="E:/photo2/face"+str(count)+".jpg"
            cv2.imwrite(filename,crop)
            print (count)
            FaceCor=FaceCor+[j[0][k]]
        print (FaceCor)
        if(FaceCor!=[]):
            picNum=0
            RandCor=RandomClipPhoto(FaceCor,img.shape[1],img.shape[0])
            for w in range(len(RandCor)):
                if(picNum==4):
                    break
                picNum+=1
                crop2=img[RandCor[w][1]:RandCor[w][1]+RandCor[w][3],RandCor[w][0]:RandCor[w][0]+RandCor[w][2]]
                filename2="E:/random2/noice"+str(noicecount)+".jpg"
                noicecount+=1
                cv2.imwrite(filename2,crop2)
        TIMES+=1# the processing image num
    log.write(i[0][0]+"  finish")
    log.close()