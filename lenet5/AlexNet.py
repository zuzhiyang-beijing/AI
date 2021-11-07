from keras .models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.datasets import cifar10

def AlexNet_module():
    net = Sequential()
    net.add(Conv2D(filters=96,kernel_size=11,strides=4,padding="valid",activation="relu",input_shape=(224,224,3)))
    net.add(MaxPooling2D(pool_size=3,strides=2))
    net.add(Conv2D(filters=256,kernel_size=5,padding="same",activation="relu"))
    net.add(MaxPooling2D(pool_size=3,strides=2))
    net.add(Conv2D(filters=384,kernel_size=3,padding="same",activation="relu"))
    net.add(Conv2D(filters=384,kernel_size=3,padding="same",activation="relu"))
    net.add(Conv2D(filters=256,kernel_size=3,padding="same",activation="relu"))
    net.add(MaxPooling2D(pool_size=3,strides=2))
    net.add(Flatten())
    net.add(Dense(4096,activation="relu"))
    net.add(Dropout(0.5))
    net.add(Dense(4096,activation="relu"))
    net.add(Dropout(0.5))
    net.add(Dense(10,activation="softmax"))
    net.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
    net.summary()
    return net

if __name__ == '__main__':
    net = AlexNet_module()

    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    plt.imshow(x_train[0])
    plt.show()
    X_TRAIN = np.zeros((x_train.shape[0],224,224,x_train.shape[3]))
    X_TEST = np.zeros((x_test.shape[0],224,224,x_test.shape[3]))
    print(X_TRAIN.shape,X_TEST.shape)
    #由于alexnet的输入参数是224*224，cifar10的数据集是32*32的，所以需要将其拉伸到224*224，如果是用imagenet的数据集就不存在这个问题
    for i in range(x_train.shape[0]):
        X_TRAIN[i] = cv2.resize(x_train[i],(224,224),interpolation=cv2.INTER_NEAREST)
    for i in range(x_test.shape[0]):
        X_TEST[i] = cv2.resize(x_test[i],(224,224),interpolation=cv2.INTER_NEAREST)
    t_train = keras.utils.to_categorical(t_train,10)
    t_test = keras.utils.to_categorical(t_test,10)
    print(x_train.shape,t_train.shape)
    net.fit(X_TRAIN,t_train,epochs=10, verbose=1)

    loss,accuracy = net.evaluate(X_TEST,t_test)
