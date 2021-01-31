from keras.models import load_model
import cv2
if __name__ == '__main__':
    model = load_model("/data/catdog/model/cat_dog.h5")

    print(model.summary())
    categories = ["dog","cat"]
    imgpath = "/data/dogs-vs-cats/1.jpg"
    img = cv2.imread(imgpath,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(100,100))
    img = img.reshape(1,100,100,3)
    result = model.predict_classes(img)

    print(categories[result[0]])
