import os
import cv2
import random
import numpy as np
import pickle
dataDir = "/data/dogs-vs-cats/train/train"
categories = ["dog","cat"]
img_size = 100
channel = 3

def createDataset():
    dataset = []
    x = []
    y = []
    for class_name in os.listdir(dataDir):
        try:
            indexname = class_name.index(".")
            if indexname == 0:
                continue
            name = class_name[0:indexname]
            index = categories.index(name)
            img = cv2.imread(os.path.join(dataDir,class_name),cv2.IMREAD_COLOR)
            img = cv2.resize(img,(img_size,img_size))
            dataset.append((img,index))
        except Exception as e:
            pass
    random.shuffle(dataset)

    for features,label in dataset:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1,img_size,img_size,channel)
    y = np.array(y)
    with open("/data/catdog/x.pickle","wb") as file:
        pickle.dump(x,file)
    with open("/data/catdog/y.pickle","wb") as file:
        pickle.dump(y,file)



if __name__ == '__main__':
    createDataset()