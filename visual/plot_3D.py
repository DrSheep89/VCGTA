import os
import csv
from pyquaternion import Quaternion
from computeAngle import *

import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_prepare(userId, timePath=r'../dataset/timestamp/1-2-Front.txt', videoId=1):
    Userdata = []
    UserFile = f'D:/VR_project/LiveDeep_All/vr-dataset/Experiment_1/{userId}/video_{videoId}.csv'

    t_list = np.loadtxt(timePath)
    with open(UserFile) as csvFile:
        csv_reader = csv.reader(csvFile)
        next(csv_reader)
        i, t_len = 0, len(t_list)
        for row in csv_reader:
            if float(row[1]) >= round(t_list[i], 3):
                v0 = [0, 0, 1]
                q = Quaternion([float(row[5]), -float(row[4]), float(row[3]), -float(row[2])])
                Userdata.append(q.rotate(v0))
                i += 1
                if i == t_len:
                    break
    Userdata = np.array(Userdata)
    return Userdata, t_list

def get_x_y(userData, H=90, W=160):
    x, y = [], []
    if userData is None:
        userData = []
    for idx, v in enumerate(userData):
        theta, phi = vector_to_ang(v)
        hi, wi = ang_to_geoxy(theta, phi, H, W)
        x.append(wi)
        y.append(hi)
    return x, y

# 设置图例字号
mpl.rcParams['legend.fontsize'] = 10

# 设置三维图形模式
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for userId in range(1, 2):
    userData, t = data_prepare(userId)
    x, y = get_x_y(userData)

    ax.plot(t, x, y, label=f'user{userId} view trajectory')


ax.set_xlabel('Time (s)')
ax.set_ylabel('X')
ax.set_zlabel('Y')

ax.legend(loc='upper left')
plt.savefig('view.svg', format='svg', dpi=200)
plt.show()