import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import  euclidean 
from tqdm._tqdm_notebook import tqdm

def distance(a,b):
    dist = sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5
    return dist 
    #return euclidean(a,b)

k = int(input("k를 입력해주시길 바랍니다."))
if torch.cuda.is_available():
    device = 'cuda'

else:
    device = 'cpu'

PATH = '/home/sumin/Documents/robot_vision/image'
images = os.listdir(PATH)
print('어떤 이미지를 출력하시겠습니까?')
for cnt, name in enumerate(images):
    print(cnt, name)
cnt = int(input())
image = cv2.imread(f'{PATH}/{images[cnt]}')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow('image',image)
#cv2.waitKey(0)
plt.figure(figsize=(10,10))
plt.title(f'{images[cnt]}')
plt.imshow(image)
plt.show()

#image = cv2.imread(f'{PATH}/{images[cnt]}')

result_image = np.zeros_like(image)
image =torch.from_numpy(image)
data = torch.reshape(image, (-1,3)).type(torch.float32)

center_b=torch.randint(0,255,(k,))
center_g=torch.randint(0,255,(k,))
center_r=torch.randint(0,255,(k,))
center = torch.Tensor(list(zip(center_b,center_g,center_r)))
center_old = torch.zeros_like(center)
labels = torch.zeros(len(data[:,0]))
error = torch.zeros(k)

for i in range(len(error)):
    error[i] = i+1

while (error.all() != 0):
    for i in range(len(data[:,0])):
        distances = torch.zeros(k)
        for j in range(k):
            distances[j] = distance(data[i], center[j])
        cluster = torch.argmin(distances)
        labels[i] = cluster 

    center_old = center
    
   
    center = center.numpy()

    for i in range(k):
        
        for j in range(len(data)):
            points = [] 
            if labels[j] ==i:
                points.append(data[j])
        #if points:
        #    pt = torch.stack(points)
        #else:
        #    pt = torch.tensor([])
        #
        #print(points)
        #pt = pt.numpy()
        #import pdb; pdb.set_trace();
        for j in range(len(points)):
            points[j] = points[j].tolist()
        
        
        
        if len(points) >1:
            center[i] =np.mean(points , axis = 0)
        
    print(center)
    
    for i in range(k):
        error[i] = distance(center_old[i],center[i])

    #center = center.tolist()
    #labels = labels.tolist()
for i in range(len(labels)):
    y = int(i/image.shape[1])
    x = i%image.shape[1]
    result_image[y][x] = (int(center[int(labels[i])][0]),int(center[int(labels[i])][1]),int(center[int(labels[i])][2]))

#print((int(center[int(labels[i])][0]),int(center[int(labels[i])][1]),int(center[int(labels[i])][2])))
#result_image =cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
#cv2.imshow("Result", result_image)
#cv2.waitKey(0)

        