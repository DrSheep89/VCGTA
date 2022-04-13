import copy
import os
import csv
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
from pyquaternion import Quaternion
from Arguments import get_args
from ConvLSTM import ConvLSTMNet
from SphereConvLSTM import SCLNet
from computeAngle import *

def index_to_xyz(pre_image: np.ndarray) -> ():
    H, W = pre_image.shape
    pos = np.argmax(pre_image)
    x, y = divmod(pos, W)   # pre_image: the index of max number
    point = [(H-x-1) % H, (W-y-1) % W]

    phi = np.arcsin(1 - 2/H * point[0])
    temp = 360 / W * point[1]
    theta = 360 - temp

    dx = np.cos(phi/180.0 * np.pi) * np.cos(theta/180.0 * np.pi)
    dy = np.sin(phi/180.0 * np.pi)
    dz = np.cos(phi/180.0 * np.pi) * np.sin(theta/180.0 * np.pi)

    return [dx, dy, dz]


def data_prepare(videoId, userId, t_list):
    Userdata = []
    UserFile = f'D:/VR_project/LiveDeep_All/vr-dataset/Experiment_1/{userId}/video_{videoId}.csv'

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
    return Userdata


def create_fixation_map(_X, _y, _idx, H, W):
    v = _y[_idx]
    theta, phi  = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H-hi-1, W-wi-1] = 1
    return result


def create_sal_fix(saliency_maps, time_path, videoId, userId):
    t_list = np.loadtxt(time_path)

    saliency_maps = de_interpolate(saliency_maps, len(saliency_maps))
    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    saliency_maps = mmscaler.fit_transform(saliency_maps.ravel().reshape(-1, 1)).reshape(saliency_maps.shape)

    N, H, W = saliency_maps.shape
    series = data_prepare(videoId, userId, t_list)

    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    fixation_map = np.array([create_fixation_map(None, series, idx, H, W) for idx,_ in enumerate(series)])
    headmap = np.array(
        [cv2.GaussianBlur(item, (args.blur_size_width, args.blur_size_high), 0) for item in fixation_map])
    fixation_maps = mmscaler.fit_transform(headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)

    return saliency_maps, fixation_maps

class FoVDataset(Dataset):
    def __init__(self, saliency_maps,  fixation_maps, frame_path, predict_time, transform=None):
        self.saliency_maps = saliency_maps
        self.fixation_maps = fixation_maps
        self.frame_path = frame_path
        self.predict_time = predict_time
        self.transform = transform
        self.length = len(saliency_maps)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frame = plt.imread(self.frame_path + f'/{idx+1}.jpg')
        if idx < self.length - self.predict_time:
            saliency = self.saliency_maps[idx]
            fixation = self.fixation_maps[idx]
            predict_fixation = self.fixation_maps[idx + self.predict_time]

        else:
            saliency = self.saliency_maps[idx]
            fixation = self.fixation_maps[idx]
            predict_fixation = self.fixation_maps[idx]

        if self.transform:
            frame = torch.from_numpy(de_interpolate_frame(frame))
            saliency = self.transform(saliency)
            fixation = self.transform(fixation)
            predict_fixation = self.transform(predict_fixation)

        return frame, saliency, fixation, predict_fixation


def sequentialData(dataset, idx, windows):
    """todo: pay attention to setting input_size = 5

    :param dataset: fov_dataset
    :param idx: index
    :param windows: predict window
    :return: mix_sal_fix, pre_fix
    """
    pre_fix_list = []
    mix_list = []
    for j in range(windows):
        frame, sal, fix, pre_fix = dataset[idx + j]
        pre_fix_list.append(pre_fix)
        mix_data = torch.cat([frame.permute(2,0,1), sal, fix], dim=0)
        mix_list.append(mix_data)

    pre_fix_output = torch.stack(pre_fix_list, dim=0)
    mix_output = torch.stack(mix_list, dim=0)

    pre_fix_output = torch.unsqueeze(pre_fix_output, 0)
    mix_output = torch.unsqueeze(mix_output, 0)

    return mix_output, pre_fix_output

def train(dataset, frameId):
    loss = 0
    print(f"Train frameId={frameId}~{frameId+args.windows}...", end='\t')
    for epoch in range(args.epochs):

        inputs, labels = sequentialData(dataset, frameId, args.windows)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        predict = net(inputs)

        loss = loss_function(predict, labels[:,args.windows//2,:,:,:])
        loss.backward()
        optimizer.step()

        # scheduler.step(loss)    # 动态调整学习率
        if (epoch+1) % 2 == 0:
            print('\rTrain frameId=[%d]~[%d]...[%3d]/[%d] loss: %.6f' %
                  (frameId, frameId + args.windows, epoch + 1, args.epochs, loss.item()), end='')
            # print('[%d/%d] loss: %.6f' % (epoch+1, args.epochs, loss.item()))

    return loss.item()

def test(dataset, frameId, startTime):
    global good_test_frame, total_test_frame

    inputs, labels = sequentialData(dataset, frameId, args.windows)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)

        predict = net(inputs)

        for j in range(args.windows-1, args.windows):
            try:
                pre_image = predict[0, 0].detach().cpu().numpy()
                real_image = labels[0, j, 0].detach().cpu().numpy()
            except IndexError:
                print('j=', j, predict.shape)

            # predictFoV.append(index_to_xyz(pre_image))
            # actualFoV.append(index_to_xyz(real_image))

            pre_image[pre_image < args.threshold] = 0
            pre_image[pre_image > args.threshold] = 1
            real_image[real_image < args.threshold] = 0
            real_image[real_image > args.threshold] = 1

            if args.showImage:
                plt.imshow(pre_image)
                plt.axis('off')
                plt.title(f'pre_image[{frameId+j}]')
                plt.show()

                plt.imshow(real_image)
                plt.axis('off')
                plt.title(f'real_image[{frameId+j}]')
                plt.show()

            fit = sum(map(sum, (pre_image + real_image) != 1))  # 统计预测的map和实际的map相同状态的块有多少
            mistake = pre_image.size - fit
            fetch = sum(map(sum, (pre_image == 1)))  # 统计预测的map中块的数量：用于带宽计算？
            need = sum(map(sum, (real_image == 1)))  # 统计实际的map中为1的块数量
            right = sum(map(sum, (pre_image + real_image) > 1))  # 统计预测的map和真实的map的同为1的块数量
            wrong = fetch - right
            if fetch == 0:
                print("\nfetch == 0", end='')

            eps = 1e-3
            tileAccuracy = round(fit / real_image.size, 4)
            recall = round((right + eps) / (need + eps), 4)
            precision = round((right + eps) / (fetch + eps), 4)
            if recall >= thres_recall:
                good_test_frame += 1
            total_test_frame += 1
            metrics = [frameId+j, fit, mistake, fetch, need, right, wrong, tileAccuracy, recall, precision]

            tileAccList.append(tileAccuracy)
            recallList.append(recall)
            precisionList.append(precision)
            logWriter.writerow(metrics)

    endTime = time.time()
    timeList.append(round(endTime - startTime, 3))
    print(f'\nTest frameId={frameId}~{frameId+args.windows} finished, time: {round(endTime - startTime, 3)}')

if __name__ == '__main__':
    # load the settings
    args = get_args()

    net = ConvLSTMNet(input_dim=args.input_size, hidden_dim=args.hidden_size,
                      kernel_size=(5, 5), num_layers=args.num_layers, batch_first=True)
    # net = SCLNet(input_dim=args.input_size, hidden_dim=args.hidden_size,
    #                   kernel_size=(3, 3), num_layers=args.num_layers, batch_first=True)
    print("Total number of parameters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    if torch.cuda.is_available():
        net = net.cuda()

    loss_function = nn.MSELoss(reduction='mean')


    nameDict = {0: 'Conan1', 1: 'Skiing', 2: 'Alien', 3: 'Conan2', 4: 'Surfing',
                5: 'War', 6: 'Cooking', 7: 'Football', 8: 'Rhinos'}
    nameList = ['1-1-Conan Gore Fly', '1-2-Front', '1-3-360 Google Spotlight Stories_ HELP',
               '1-4-Conan Weird Al', '1-5-TahitiSurf', '1-6-Falluja',
               '1-7-Cooking Battle', '1-8-Football', '1-9-Rhinos']

    timeBasePath = './dataset/timestamp/'
    salBasePath = './dataset/saliency/'
    frameBasePath = './dataset/'

    idx = 0
    for videoId in [1, 2, 6]:
        print('#'*30, nameDict[videoId], '#'*30)
        video_time = 0
        for userId in range(1, 2): # 用户ID
            print('*' * 20, f'Test video_{videoId}...for user={userId}', '*' * 20)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, args.epochs],
            #                                                  gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

            saliencyPath = salBasePath + f"{nameList[videoId]}.npy"
            timePath = timeBasePath + f"{nameList[videoId]}.txt"
            framePath = frameBasePath + f'{nameList[videoId]}'

            saliency_array = np.load(saliencyPath, allow_pickle=True)
            total_frame = len(saliency_array)

            saliency_maps, fixation_maps = create_sal_fix(saliency_array, timePath, videoId, userId)

            transform = transforms.Compose([transforms.ToTensor()])

            fov_dataset = FoVDataset(saliency_maps, fixation_maps, framePath, args.windows, transform)

            log_path = args.log_path + f'/{nameDict[videoId]}'
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logName = log_path + f"/{userId}.csv"

            predictFoV = []
            actualFoV = []

            with open(logName, 'w', newline='') as f:
                logWriter = csv.writer(f, dialect='excel')
                logWriter.writerow(['FrameId', 'Match', 'Mistake', 'PredictTile', 'RealTile',
                                    'RightTile', 'RedundantTile', 'Acc', 'Recall', 'Precise'])

                total_test_frame = 0
                good_test_frame = 0
                tileAccList, recallList, precisionList = [], [], []
                thres_recall = 0.6
                start_time = time.time()
                timeList = []

                net.initialize_weight()
                for frame_id in range(args.windows, total_frame-2*args.windows, args.windows//2):
                    window_start_time = time.time()

                    # online train
                    loss = train(fov_dataset, frame_id)
                    if loss > 0.0001:
                        net.initialize_weight()
                    test(fov_dataset, frame_id+args.windows, window_start_time)           # 然后预测下一段

                end_time = time.time()
                user_time = round(end_time - start_time, 5)
                video_time += end_time - start_time
                logWriter.writerows([
                    ['total_time', user_time],
                    ['average_time', np.mean(timeList)],
                    ['total_test_frame', total_test_frame],
                    ['good_test_frame', good_test_frame],
                    ['threshold_recall', thres_recall],
                    ['FrameAccuracy', round(good_test_frame / total_test_frame, 4)],
                    ['AverageTileAccuracy', np.mean(tileAccList)],
                    ['AverageRecall', np.mean(recallList)],
                    ['AveragePrecision', np.mean(precisionList)]
                ])

            # np.savetxt("./predictFoV.txt", np.array(predictFoV))
            # np.savetxt("./actualFoV.txt", np.array(actualFoV))
            print('*' * 20, f'Test video={nameDict[videoId]} for userId={userId} finished, time={user_time}s', '*' * 20)
            # break
        print(f'Test video_{videoId} finished, time={round(video_time, 4)}s')

