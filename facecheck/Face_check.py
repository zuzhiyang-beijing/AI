import face_recognition as fr
import cv2
import os
import pickle
import numpy as np


def identify_faces(img_path):
    img = fr.load_image_file(img_path)
    face_loacation = fr.face_locations(img,model='hog')
    imgencodings = fr.face_encodings(img,face_loacation)
    pickle_in = open("/data/face_pickle/name_encoding.pickle","rb")
    known_faceEncoing = pickle.load(pickle_in)
    names = list(known_faceEncoing.keys())
    encodings =list(known_faceEncoing.values())

    face_names = []
    for imgencodin in imgencodings:
        distances = fr.face_distance(encodings,imgencodin)

        results = fr.compare_faces(encodings,imgencodin,tolerance=0.75)

        best_match_index = np.argmin(distances);
        if results[best_match_index]:
            name = names[best_match_index];
            face_names.append(name)
    return  img,face_loacation,face_names

def getresut(img_path):

    img,face_location,face_names = identify_faces(img_path)

    if len(face_names) != 0:
        for (top,right,bottom,left),name in zip(face_location,face_names):
            cv2.rectangle(img,(left-20,top-20),(right+20,bottom+20),(255,0,0),2)
            cv2.rectangle(img,(left-20,bottom-20),(right+20,bottom+20),(255,0,0),cv2.FONT_HERSHEY_DUPLEX)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img,name,(left-20,bottom+15),font,1.0,(255,255,255),2)
        savename = "_".join(face_names)
        cv2.imwrite("/data/faceimg/2.jpg",img)
    else:
        print("没这个人")

if __name__ == '__main__':
    dir = "/data/faceimg/checkrepo/"
    imgfiles = os.listdir(dir)
    name_encodings = {}
    for img_name in imgfiles:
        if img_name.endswith(".JPG") == False:continue
        face = fr.load_image_file(os.path.join(dir,img_name))
        encoding = fr.face_encodings(face)[0]
        name = img_name.split(".")[0]
        name_encodings[name] = encoding
    pickle_out = open("/data/face_pickle/name_encoding.pickle","wb")
    pickle.dump(name_encodings,pickle_out)
    pickle_out.close()
    getresut("/data/faceimg/1.JPG")