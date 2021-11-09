from keras.layers import Conv2D,MaxPooling2D,Input,Dense
from keras.models import Model,Sequential

"""
定义vgg块
"""
def vgg_block(num_convs, n_filters,net):
    layers = []
    for _ in range(num_convs):
        net.add(Conv2D(filters=n_filters,kernel_size=(3*3),padding="same",activation="relu"))
    net.add(MaxPooling2D(pool_size=(2*2),strides=2,padding="same"))
    return layers


def vgg_Model():
    net = Sequential()
    input = Input(shape=(224,224,3),name="input")
    net.add(input)
    vgg_block(1,64,net)
    vgg_block(1,128,net)
    vgg_block(2,256,net)
    vgg_block(2,512,net)
    vgg_block(2,512,net)

    net.add(Dense(4096,activation="relu"))
    net.add(Dense(4096,activation="relu"))
    net.add(Dense(10,activation="softmax"))
    net.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
    net.summary()
    return net

if __name__ == '__main__':
    vgg_Model()