import face_recognition as fr
import cv2

if __name__ == '__main__':
    img_path = "/data/faceimg/1.JPG"
    img = fr.load_image_file(img_path)
    face_location = fr.face_locations(img,model='hog')
    landmarks = fr.face_landmarks(img)
    print(landmarks)
    n_face = len(face_location)
    print(n_face)

    for fc in face_location:
        top,right,bottom,left = fc
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
    cv2.imwrite("/data/faceimg/1_location.jpg",img)