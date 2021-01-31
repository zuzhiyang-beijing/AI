首先根据原始文件生成特征文件
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
    
然后根据传入照片进行人脸比对
