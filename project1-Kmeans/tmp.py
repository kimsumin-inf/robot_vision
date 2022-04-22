import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from copy import deepcopy

def distance(a, b):
    return sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5



save_dir = '/home/sumin/Documents/robot_vision/image'
lists = os.listdir(save_dir)

image =[]
for k in lists:
    save = save_dir+'/'+k
    img = cv2.imread(save)
    image.append(img)



k=8



for i in range(len(image)):
    data = image[i].reshape((-1, 3)).astype(np.float32)
    img_copy = image[i].copy()

    centroids_b = np.random.randint(0, 255, k)
    centroids_g = np.random.randint(0, 255, k)
    centroids_r = np.random.randint(0, 255, k)
    centroids = zip(centroids_b, centroids_g, centroids_r)
    centroids = list(centroids)

    centroids_old = deepcopy(centroids)  # 제일 처음 centroids_old는 0으로 초기화 해줍니다
    labels = np.zeros(len(data[:,0]))
    error = np.zeros(k)

    for j in range(len(error)):
        error[j]=j+1
    # STEP 4: error가 0에 수렴할 때 까지 2 ~ 3 단계를 반복합니다
    while (error.all() != 0):
        # STEP 2: 가까운 centroids에 데이터를 할당합니다
        for l in range(len(data[:,0])):
            distances = np.zeros(k)  # 초기 거리는 모두 0으로 초기화 해줍니다
            for j in range(k):
                distances[j] = distance(data[l], centroids[j])
            cluster = np.argmin(distances)  # np.argmin은 가장 작은 값의 index를 반환합니다
            labels[l] = cluster
        # Step 3: centroids를 업데이트 시켜줍니다
        centroids_old = deepcopy(centroids)
        for l in range(k):
            # 각 그룹에 속한 데이터들만 골라 points에 저장합니다
            points = [data[j] for j in range(len(data)) if labels[j] == l]
            # points의 각 feature, 즉 각 좌표의 평균 지점을 centroid로 지정합니다
            print(len(points))
            if len(points)>1:
                centroids[l] = np.mean(points, axis=0)
        # 새롭게 centroids를 업데이트 했으니 error를 다시 계산합니다
        for j in range(k):
            error[j] = distance(centroids_old[j], centroids[j])

    for j in range(len(labels)):
        y = int(j/image[i].shape[1])
        x = j % image[i].shape[1]

        img_copy[y][x] = (int(centroids[int(labels[j])][0]),int(centroids[int(labels[j])][1]),int(centroids[int(labels[j])][2]))

    cv2.imshow("clustering", img_copy)
    cv2.waitKey(0)