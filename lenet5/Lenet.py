from keras .models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.optimizers import SGD
from dataset.mnist import load_mnist
import numpy as np
def Lenet5_model():
    net = Sequential()
    net.add(Conv2D(filters=6, kernel_size=(5,5), padding="valid",activation='sigmoid',input_shape=(28,28,1)))
    net.add(AveragePooling2D(pool_size=(2,2), strides=2))
    net.add(Conv2D(filters=16, kernel_size=(5,5), padding="valid",activation='sigmoid'))
    net.add(AveragePooling2D(pool_size=(2,2), strides=2))
    net.add(Flatten())
    # 默认情况下，“Dense” 会自动将形状为（批量大小，通道数，高度，宽度）的输入，
    # 转换为形状为（批量大小，通道数*高度*宽度）的输入
    net.add(Dense(120,activation='sigmoid'))
    net.add(Dense(84, activation='sigmoid'))
    net.add(Dense(10,activation='softmax'))

    #loss = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

    net.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
    net.summary()
    return net


if __name__ == '__main__':
    net = Lenet5_model()
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)
    #x_train = np.expand_dims(x_train, axis=0)
    #t_train =  np.expand_dims(t_train, axis=0)
    print(x_train.shape,t_train.shape)
    #训练
    net.fit(x_train, t_train, epochs=10, verbose=1)
    ##测试准确率和损失函数
    loss,accuracy = net.evaluate(x_test,t_test)
