import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
PATH = '/home/sumin/Documents/robot_vision/SIFT'
images = os.listdir(PATH)
images = sorted(images)
print('어떤 이미지를 출력하시겠습니까?')
for cnt, name in enumerate(images):
    print(cnt, name)
cnt = int(input())
image = cv2.imread(f'{PATH}/{images[cnt]}')
#matplotlib 출력시
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Make a Scale Space 
Octave =[]

for i in range (4):
    img = []
    tmp = cv2.resize(image,(int(image.shape[1]*2**(1-i)),int(image.shape[0]*2**(1-i))))
    for j in range(5):
        img.append(cv2.GaussianBlur(tmp, (0,0),2**(j/2)))
    Octave.append(img)

# DoG Calculation
DoG=[] 
for i in range(4):
    tmp =[]
    for j in range(4):
        tmp.append(Octave[i][j] - Octave[i][j+1])
    DoG.append(tmp)
len(DoG)

# Keypoint 찾기

from tqdm.notebook import tqdm 
Key_Points = [] 
for j in range(4):
    tmp = [] 
    for i in tqdm(range(1,3)):
        key = np.zeros_like(DoG[j][0])
        print(j, i)  
        for y in range(1, len(key[:-1,0])):
            for x in range(1, len(key[0,:-1])):
                lis =[DoG[j][i][y-1][x-1],DoG[j][i][y-1][x],DoG[j][i][y-1][x+1],DoG[j][i][y][x-1],DoG[j][i][y][x],DoG[j][i][y][x+1],DoG[j][i][y+1][x-1],DoG[j][i][y+1][x],DoG[j][i][y+1][x+1],
                        DoG[j][i-1][y-1][x-1],DoG[j][i-1][y-1][x],DoG[j][i-1][y-1][x+1],DoG[j][i-1][y][x-1],DoG[j][i-1][y][x],DoG[j][i-1][y][x+1],DoG[j][i-1][y+1][x-1],DoG[j][i-1][y+1][x],DoG[j][i-1][y+1][x+1],
                        DoG[j][i+1][y-1][x-1],DoG[j][i+1][y-1][x],DoG[j][i+1][y-1][x+1],DoG[j][i+1][y][x-1],DoG[j][i+1][y][x],DoG[j][i+1][y][x+1],DoG[j][i+1][y+1][x-1],DoG[j][i+1][y+1][x],DoG[j][i+1][y+1][x+1]]
                #import pdb;pdb.set_trace()
                if DoG[j][i][y][x].all() == np.min(lis) or DoG[j][i][y][x].all() == np.max(lis):
                    key[y][x] = DoG[j][i][y][x]
        tmp.append(key)
    Key_Points.append(tmp)
print(len(Key_Points))

## 나쁜 Key Point 제거 
Good_Key_Point =[]
threshold = 100

for j in range(4):
    tmp = [] 
    for i in tqdm(range(2)):
        key = np.zeros_like(Key_Points[j][0])
        print(j, i)  
        for y in range( key.shape[0]):
            for x in range( key.shape[1]):
                if Key_Points[j][i][y,x] < threshold:    
                    key[y,x] = 0
                else :
                    key[y,x] = Key_Points[j][i][y,x] 

        tmp.append(key)
    Good_Key_Point.append(tmp)


for i in range(len(Good_Key_Point)):
    for j in range(len(Good_Key_Point[i])):
        for y in range(Good_Key_Point[i][j].shape[0]):
            for x in range(Good_Key_Point[i][j].shape[1]):
                if not (Good_Key_Point[i][j][y,x]==0):
                    #Octave[i][j]=cv2.cvtColor(Octave[i][j],cv2.COLOR_GRAY2BGR)
                    Octave[i][j]=cv2.circle(Octave[i][j],(x,y),2, color=255)

        cv2.imshow(f"img",Octave[i][j])
        cv2.waitKey(0)