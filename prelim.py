import numpy as np
from keras.preprocessing import image

def load_seq(filename, B, N, F, D):
    with open(filename) as file:
        text = file.read()
        text = text.splitlines()
    dat = np.zeros((len(text), 10))
    for i in range(0,len(text)):
        l = text[i].split(',')
        for j in range(0,len(l)):
            dat[i][j] = float(l[j])
    
    data = np.full((B, N, F, D), np.nan)
    label = np.full((B, N, F), np.nan, dtype=int)
    num_targets = np.zeros((B, F), dtype=int)
    i = 0
    k = 0
    while dat[i][0] <= B * F:
        b = int((dat[i][0]-1) // F)
        r = int((dat[i][0]-1) % F)
        data[b][k][r][:] = dat[i][2:6]
        label[b][k][r] = int(dat[i][1])
        i += 1
        k += 1
        while k >= N and dat[i][0] == dat[i-1][0]:
            i += 1
        if i > 0 and dat[i][0] != dat[i-1][0]:
            num_targets[b][r] = k
            k = 0
    return (data, label)
    
def loadTable(filename, B, M, N, F, D):
    (detections, dummy) = load_seq(filename + "det/det.txt", B, N, F, D)
    img = image.load_img(filename + "det/000001-acf.jpg")
    sz = img.size
    detections[0,:,:,0] = detections[0,:,:,0]/sz[0] - 0.5
    detections[0,:,:,1] = detections[0,:,:,1]/sz[1] - 0.5
    detections[0,:,:,2] = detections[0,:,:,2]/sz[0] - 0.5
    detections[0,:,:,3] = detections[0,:,:,3]/sz[1] - 0.5
    (tracks, labels) = load_seq(filename + "gt/gt.txt", B, M, F, D)
    tracks[0,:,:,0] = tracks[0,:,:,0]/sz[0] - 0.5
    tracks[0,:,:,1] = tracks[0,:,:,1]/sz[1] - 0.5
    tracks[0,:,:,2] = tracks[0,:,:,2]/sz[0] - 0.5
    tracks[0,:,:,3] = tracks[0,:,:,3]/sz[1] - 0.5
    return (tracks, detections, labels)

# Example: traindata = loadTable("/home/raj/Documents/project/amilan-rnntracking-64d477848af3/src/2DMOT2015Labels/train/ADL-Rundle-8/", 1, 20, 20, 20, 4)

import os
import PIL
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, SimpleRNN, LSTM
from keras import backend as K
from keras.models import Model

def getDA(N, M, hidden_units):
    a = Input(shape=(N,M))
    b = LSTM(hidden_units, return_sequences=True, activation='tanh')(a)
    c = Dense(M, activation='softmax')(b)
    model = Model(input=a, output=c)
    return model

def getDistMat(tracks, detections, F, N, M):
    label = np.zeros((F, N, M))
    Ct = np.full((F, N, M), 1000, dtype=float)
    for t in range(0,F):
        for i in range(0,N):
            for j in range(0,M):
                dist = np.linalg.norm(tracks[0,i,t,:]-detections[0,j,t,:])
                if not np.isnan(dist) and not np.isinf(dist):
                    Ct[t, i, j] = dist
            ind = np.argmin(Ct[t,i,:])
            label[t,i,ind] = 1
    return (Ct, label)

N = 1
M = 20
F = 50
B = 1
D = 4
DA = getDA(N, M, 50)

(train_tracks, train_det, dummy) = loadTable("/home/raj/Documents/project/amilan-rnntracking-64d477848af3/src/2DMOT2015Labels/train/ADL-Rundle-8/", B, N, M, F, D)

(test_tracks, test_det, dummy) = loadTable("/home/raj/Documents/project/amilan-rnntracking-64d477848af3/src/2DMOT2015Labels/train/ADL-Rundle-6/", B, N, M, F, D)

(train_x, train_y) = getDistMat(train_tracks, train_det, F, N, M)
(test_x, test_y) = getDistMat(test_tracks, test_det, 20, N, M)

DA.compile( loss="categorical_crossentropy", optimizer="Nadam", metrics = ['categorical_accuracy'])
history1 = DA.fit(train_x, train_y, batch_size=5, nb_epoch=30, validation_data=(test_x, test_y))

pred = DA.predict(train_x)
da1 = np.argmax(pred, axis=2)
true_det = np.zeros((F, D), dtype=float)
gt = np.zeros((F, D), dtype=float)
error = np.zeros((F,1), dtype=float)
for i in range(0,F):
    true_det[i,:] = np.multiply(train_det[0,da1[i],i,:]+0.5, np.array((1920, 1080, 1920, 1080)))
    gt[i,:] = np.multiply(train_tracks[0,0,i,:]+0.5, np.array((1920, 1080, 1920, 1080)))
    error[i] = np.linalg.norm(gt[i,:] - true_det[i,:])
