from keras.layers import Conv2D,MaxPooling2D,Input,Dense,Dropout,GlobalAveragePooling2D,Flatten
from keras.models import Model,Sequential

def nin_block(num_channels,kernel_size,strides,padding):
    blk = Sequential()
    blk.add(Conv2D(filters=num_channels,kernel_size=kernel_size,strides=strides,padding=padding,activation="relu"))
    blk.add(Conv2D(filters=num_channels,kernel_size=(1,1),activation="relu"))
    blk.add(Conv2D(filters=num_channels,kernel_size=(1,1),activation="relu"))
    return blk


def nin_model():
    net = Sequential()
    input = Input(shape=(224,224,3),name="input")
    net.add(input)
    net.add(nin_block(96, kernel_size=11, strides=4, padding='valid'))
    net.add(MaxPooling2D(pool_size=3, strides=2))
    net.add(nin_block(256, kernel_size=5, strides=1, padding='same'))
    net.add(MaxPooling2D(pool_size=3, strides=2))
    net.add(nin_block(384, kernel_size=3, strides=1, padding='same'))
    net.add(MaxPooling2D(pool_size=3, strides=2))
    net.add(Dropout(0.5))
    net.add(nin_block(10, kernel_size=3, strides=1, padding='same'))
    net.add(GlobalAveragePooling2D())
    net.add(Flatten())
    net.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
    net.summary()
    return net

if __name__ == '__main__':
    nin_model()