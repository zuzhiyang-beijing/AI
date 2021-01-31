import numpy as np
import matplotlib.pyplot as plt
import pylab

from keras.models import Model
from keras.layers import Input,Dense,Flatten,BatchNormalization
from keras.layers import Conv2D,Conv2DTranspose,MaxPooling2D,UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist

if __name__ == '__main__':

    (x_train,_),(x_test,_) = mnist.load_data()
    x_train = x_train/255
    x_text = x_test/255
    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))

    input_layer = Input((28,28,1))

    x = Conv2D(10,(5,5),activation="relu")(input_layer)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(20,(2,2),activation="relu")(x)
    x = MaxPooling2D((2,2))(x)

    encoding = x
    x = UpSampling2D((2,2))(encoding)
    x = Conv2DTranspose(20,(2,2),activation="relu")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(10,(5,5),activation="sigmoid")(x)
    output_layer = Conv2DTranspose(1,(3,3),activation="sigmoid")(x)
    model = Model(inputs=input_layer,outputs=output_layer)

    print(model.summary())

    model.compile(loss="binary_crossentropy",optimizer=Adam(),metrics=None)
    model.fit(x_train,x_train,batch_size=21,epochs=1,validation_data=(x_test,x_test))


    #模型使用
    plt.imshow(x_train[11].reshape(28,28))
    pylab.show()
    result = model.predict(x_train[11].reshape(1,28,28,1))
    plt.imshow(result.reshape(28,28))
    pylab.show()
