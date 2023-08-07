from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob, numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img, array_to_img
from keras.utils import to_categorical
from keras.models import load_model
"""
#각 데이터 수 늘리기
def IDG(fname):
    ImageDG = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1, fill_mode='nearest')
    img = tf.keras.preprocessing.image.load_img(fname)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    save = fname.split('/')[0] + "/" + fname.split('/')[1] + fname.split('/')[2] + fname.split('/')[3]
    for bath in ImageDG.flow(x, batch_size=1, save_to_dir=save, save_prefix='new', save_format='jpg'):
        i += 1
        if i > 7: break

folder_list = os.listdir('./archive/data/train')
fname = "./archive/data/train/"
for f in folder_list:
    fname = "./archive/data/train/" + f + "/"
    file_list = os.listdir(fname)
    for i in file_list:
        filename = fname + i
        IDG(filename)

ImageDG = ImageDataGenerator(rescale=1. / 255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1, fill_mode='nearest')
folder_list = os.listdir('./archive/data/train')
#fname = "./archive/data/train/"
for f in folder_list:
    fname = "./archive/data/train/" + f + "/"
    file_list = os.listdir(fname)
    for i in file_list:
        filename = fname + i
        img = tf.keras.preprocessing.image.load_img(fname)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        save = fname.split('/')[0] + "/" + fname.split('/')[1] + fname.split('/')[2] + fname.split('/')[3]
        for bath in ImageDG.flow(x, batch_size=1, save_to_dir=save, save_prefix='new', save_format='jpg'):
            i += 1
            if i > 7: break
"""
#Image 데이터를 학습 데이터로 변환
img_dir = "./archive/data/train/"
categories = os.listdir(img_dir)
num_classes = len(categories)

image_w = 64
image_h = 64

pixel = image_w * image_h * 3
X=[]
Y=[]

for idx, cat in enumerate(categories):
    img_dir_detail = img_dir + '/' + cat
    files = glob.glob(img_dir_detail + "/*.jpg")
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.array(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(cat, ":", f)
        except:
            print(cat, str(i), "번쨰에서 에러")

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#학습데이터 가공
X_train = X_train.astype(float)/255.0
X_test = X_test.astype(float)/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
#모델구축
with tf.device('/device:CPU:0'):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model_dir = "./model"
    model_path = model_dir + "/cloud_classify.model"

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.15, callbacks=[checkpoint, early_stopping])
    print("정확도 : %.2f" % (model2.evaluate(X_test, y_test)[1]))
"""
with tf.device('/device:CPU:0'):
    model2 = Sequential()

    model2.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))

    model2.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))

    model2.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model2.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))

    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(num_classes, activation="softmax"))

    model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model_dir = "./model2"
    model_path = model_dir + "/cloud_classify.model2"

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    print(model2.summary())

    history = model2.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.15,
                        callbacks=[checkpoint, early_stopping])

    print("정확도 : %.2f" % (model2.evaluate(X_test, y_test)[1]))

path = './archive/data/test/'
category = os.listdir('.archive/data/train')

image_w = 64
image_h = 64

pixels = image_h * image_w

X = []
filenames = []
files = glob.blob(path+'/*.*')
for f in files:
    img = Image.ope(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
prediction_test = model2.predict(X)

file_index = 0
for i in prediction_test:
    label = i.argmax()
    print("///////////////////////////////////////////////")
    print(filenames[file_index].split('\\')[-1] + "의 예측되는 구름종류 :" + category[label])
    file_index = file_index + 1