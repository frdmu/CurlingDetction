import numpy as np 
import os.path
import codecs
import cv2
import pandas as pd
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers

def load(datafile):
    fr = codecs.open(datafile, 'r', 'utf-8')
    train_list = fr.readlines()
    #print(np.array(train_list))
    dataset = []
    for line in train_list:
        tmp = line.strip().split(',')
        #print(tmp, tmp[0], tmp[1], tmp[2])
        fpath = tmp[0]
        img = cv2.imread(fpath)
        np_img = np.asarray(img, dtype="int")
        np_img = np_img.flatten() 
        np_label = np.array([int(tmp[1]), int(tmp[2])])
        np_line = np.concatenate((np_img, np_label))
        dataset.append(np_line)
    fr.close()
    return np.array(dataset) 
def load_data():
    dataset = load("train_list.txt")
    print(dataset.shape)
    df = pd.DataFrame(dataset)
    #print(type(df))
    display(df.head())
    ##################1.数据处理#####################
    df_train = df.sample(frac=0.7, random_state=0)
    df_valid = df.drop(df_train.index)
    #找到每一列的最大值和最小值，以便将数据全部转换为0-1之间的数字
    max_ = df_train.max(axis=0)
    min_ = df_train.min(axis=0)
    df_train = (df_train - min_) / (max_ - min_)
    df_valid = (df_valid - min_) / (max_ - min_)
    #X_train 为训练集的输入，X_valid 为验证集的输入，y_train为训练集的输入， y_valid为验证集的输出
    X_train = np.array(df_train.drop(df_train.columns[[3000000, 3000001]], axis=1))
    X_valid = np.array(df_valid.drop(df_valid.columns[[3000000, 3000001]], axis=1))
    y_train = np.array(df_train[[3000000, 3000001]])
    y_valid = np.array(df_valid[[3000000, 3000001]])
    return X_train, y_train, X_valid, y_valid


def neuralNetwork(X_train, y_train, X_valid, y_valid):
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    model = keras.Sequential([
    layers.BatchNormalization(input_shape=[input_shape]),	

    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),	

    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),	

    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),	

    layers.Dense(units=output_shape),
    ])

    model.compile(optimizer='adam', loss='mae', metrics=[mae])

    early_stopping = keras.callbacks.EarlyStopping(
      patience=10,
      min_delta=0.001,
      restore_best_weights=True,
    )

    history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping],
    verbose=0,# hide the output because we have so many epochs
    )


    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid = load_data()
    neuralNetwork(X_train, y_train, X_valid, y_valid)
