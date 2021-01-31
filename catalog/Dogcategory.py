from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.optimizers import Adam
import pickle

if __name__ == '__main__':
    x = []
    y = []
    with open("/data/catdog/x.pickle","rb") as file:
        x = pickle.load(file)
    with open("/data/catdog/y.pickle","rb") as file:
        y = pickle.load(file)
    x = x/255.0
    n_samples = x.shape[0]
    split = 0.1
    n_train = int(n_samples*(1-split))
    x_train = x[0:n_train,:,:,:]
    y_train = y[0:n_train]

    x_test = x[n_train:,:,:,:]
    y_test = y[n_train:]

    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=x.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.2))

    model.add(Dense(2,activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy",optimizer=Adam(),metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=10,verbose=1,batch_size=64,validation_data=(x_test,y_test))

    model.save("/data/catdog/model/cat_dog.h5")

    #model.load_model("")
