import pandas as pd
import numpy as np
import jieba as jb
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras .models import  Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import  Conv1D
from keras.layers.convolutional import  MaxPooling1D
import keras
from keras .models import load_model


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line

#停用词
# 加载停用词
stopwords = ""#stopwordslist("停用词.txt")
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='gbk').readlines()]
    return stopwords

#将单词编成整数系列
#单词-整数的映射
def create_tokenizer(lines):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

#使用tokenizer.text_to_sequences()函数来获取单词-整数编码
#使用pad_sequences 函数来为长度不够的文本进行填0操作，使所有文本长度一致
def encode_docs(tokenizer,max_length,docs):
    encoded=tokenizer.texts_to_sequences(docs)#单词-整数映射
    padded=pad_sequences(encoded,maxlen=max_length,padding='post')
    return padded


#定义神经网络模型
def define_model(vocad_size,max_length):
    model=Sequential()
    model.add(Embedding(vocad_size,100,input_length=max_length))
    model.add(Conv1D(filters=32,kernel_size=5,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
#模型训练
def model_train(vocad_size, max_length,X_train, Y_train):
    model = define_model(vocad_size, max_length)
    model.fit(X_train, Y_train, epochs=10, verbose=2)
    model.save('词向量模型分类.h5')

yd_names1 = {'0':'差评','1':'好评'}

def predict_text(text,stopwords,tokenizer,max_length,model):
    line='地理位置不错, 到哪里比较方便'
    line = remove_punctuation(text)
    line = jb.lcut(line)
    print(line)
    #cw = lambda x: list(jb.cut(x))
    #line =[text].apply(cw)
    #填充长度,注意line要加[]号括起来。
    padded=encode_docs(tokenizer,max_length,[line])
    y_predict=model.predict(padded,verbose=0)
    pred_y = np.argmax(y_predict, axis=1) #得到的是最大数值索引

    print(yd_names1[str(pred_y[0])])#获取类型文字信息

if __name__ == '__main__':
    csv='/data/preProcess/cutclean_label_corpus10000.csv'
    file_txt=pd.read_csv(csv)#[1169656 rows x 3 columns]

    file_txt=file_txt.dropna()#删除空值[1005981 rows x 2 columns]

    #print(file_txt)

    #去除标点符号
    file_txt['clean_review']=file_txt['text'].apply(remove_punctuation)
    #去除停用词
    #file_txt['cut_review']=file_txt['clean_review'].apply(lambda x:" ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
    cw = lambda x: list(jb.cut(x))
    file_txt['cut_review']=file_txt['clean_review'].apply(cw)
    tokenizer=create_tokenizer(file_txt['cut_review'])
    print(file_txt['cut_review'])
    max_length=max([len(s.split()) for s in file_txt['text']])
    print('最长词语 句子：',max_length)#126

    X_train=encode_docs(tokenizer,max_length,file_txt['cut_review'])
    y_train=file_txt['label']
    y_train=np.array(y_train)#(130583, 1)
    Y_train=keras.utils.to_categorical(y_train,2)#2分类
    #定义神经网络模型
    """
    #使用Embedding层作为第一层，需要指定词汇表大小，
    实值向量空间的额大小以及输入文档的最大长度。词汇表大小是我们词汇表中的单词总数，加上一个未知单词
    """
    vocad_size=len(tokenizer.word_index)+1
    print('词汇表大小：',vocad_size)#7896

    #model_train(vocad_size,max_length,X_train,Y_train)

    #模型评估
    from sklearn.metrics import accuracy_score
    #分类器评估
    model=load_model('词向量模型分类.h5')
    #predict_y=model.predict(X_train)
    #pred_y=np.argmax(predict_y,axis=1)
    #test_y=np.array(file_txt['label'])
    #f1=accuracy_score(pred_y,test_y)
   #print(f1)
    predict_text("分量上，一般，很不划算","",tokenizer,max_length,model)

