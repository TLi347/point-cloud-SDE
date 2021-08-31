import torch
from torch import nn

import numpy as np

side = 16
batch_size = 16

def ini_img_b(action, width, height, batch_size=16):
    action = (action*(width-1)).astype(int)
    res = np.zeros(shape=(batch_size,width,height))
    for i in range(width):
        for j in range(height):
            value = np.sqrt( (i-action[:,0])**2 + (j-action[:,1])**2) - action[:,2]
            res[:, i, j] = value
    return res

def ini_bimg(action, width, height):
    action = (action*(width-1)).astype(int)
    res = np.zeros(shape=(1, width,height))
    for i in range(width):
        for j in range(height):
            value = np.sqrt( (i-action[0])**2 + (j-action[1])**2) - action[2]
            if value < 0:
                res[:, i, j] = -1.0
            else:
                res[:, i, j] = 1.0
    return res

def ini_pri(action, length, width, height):
    action = (action*(width-1)).astype(int)
    res = np.zeros(shape=(length, width,height))
    for i in range(length):
        for j in range(width):
            for k in range(height):
                value = np.sqrt( (i-action[0])**2 + (j-action[1])**2 + (k-action[2])**2) - action[3]
                res[:, i, j, k] = value
    res = res/128
    return res

def ini_bpri(action, length, width, height):
    action = (action*(width-1)).astype(int)
    res = np.zeros(shape=(length, width,height))
    for i in range(length):
        for j in range(width):
            for k in range(height):
                value = np.sqrt( (i-action[0])**2 + (j-action[1])**2 + (k-action[2])**2) - action[3]
                if value < 0:
                    res[i, j, k] = -1.0
                else:
                    res[i, j, k] = 1.0
    return res

# width = 128
# height = 128
# action = (np.random.random(3).reshape(3))
# img = torch.from_numpy(ini_img(action,width=width, height=height))
# print(img.shape)
# print(img[:,4,5])
# import util
# coordinates, features = util.to_coordinates_and_features(img,width,is_pri=True)#[w*h, 3], [w*h, 3] 
# coordinates, features = coordinates, features
# print('train', coordinates.shape, features.shape)
# print(torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1).shape)
# coordinates = torch.cat((coordinates, torch.from_numpy(action).unsqueeze(0).repeat(coordinates.shape[0],1)),dim=1)
# print(coordinates.shape)