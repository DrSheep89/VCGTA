import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os

logPathList = ['../log_w4/CL(frame+sal+fix)/',
           '../log_w8/CL(frame+sal+fix)/',
           '../log_w12/CL(frame+sal+fix)/',
           '../log_w16/CL(frame+sal+fix)/',]

nameList = ['Alien', 'Conan1', 'Cooking', 'Skiing']

num = 24
time = []
time_std = []

for logPath in logPathList:
    timeList = []
    for idx, name in enumerate(nameList):
        for i in range(1, num+1):
            with open(logPath + f"{name}/{i}.csv", 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
                try:
                    timeList.append(round(float(rows[-8][1]), 4))
                except IndexError:
                    print("IndexError:", idx)
    time.append(timeList)
    # time.append(np.mean(timeList))
    # time_std.append(np.std(timeList))



timeList = [1, 2, 3, 4]
accList, recallList, precisionList, F1List = [], [], [], []

with open('../log/metrics.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    accList.append(np.mean([float(row) for row in rows[1][1:]]))
    accList.append(np.mean([float(row) for row in rows[5][1:]]))
    accList.append(np.mean([float(row) for row in rows[9][1:]]))
    accList.append(np.mean([float(row) for row in rows[13][1:]]))

    recallList.append(np.mean([float(row) for row in rows[2][1:]]))
    recallList.append(np.mean([float(row) for row in rows[6][1:]]))
    recallList.append(np.mean([float(row) for row in rows[10][1:]]))
    recallList.append(np.mean([float(row) for row in rows[14][1:]]))

    precisionList.append(np.mean([float(row) for row in rows[3][1:]]))
    precisionList.append(np.mean([float(row) for row in rows[7][1:]]))
    precisionList.append(np.mean([float(row) for row in rows[11][1:]]))
    precisionList.append(np.mean([float(row) for row in rows[15][1:]]))

    F1List.append(np.mean([float(row) for row in rows[3][1:]]))
    F1List.append(np.mean([float(row) for row in rows[8][1:]]))
    F1List.append(np.mean([float(row) for row in rows[12][1:]]))
    F1List.append(np.mean([float(row) for row in rows[16][1:]]))

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 24
fig = plt.figure(figsize=(16.5, 8))
# fig = plt.figure(figsize=(16, 6))

plt.subplot(121)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.plot(timeList, accList, marker='^', label='accuracy', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, recallList, marker='s', label='recall', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, precisionList, marker='d', label='precision', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, F1List, marker='o', label='F1-score', linestyle='dashed', linewidth=3.0, markersize=10)
plt.xlabel('Prediction window (s)')
plt.ylabel('Average metric')
plt.tick_params(labelsize=22)
font_dict=dict(fontsize=28,
              # color='b',
              family='Times New Roman',
              weight='bold',
              # style='italic',
              )
plt.legend(ncol=2, fontsize=21)
plt.grid(linestyle='-')
plt.title('(a) The performance of ConvLSTM(fsu)\nfor different prediction windows',
          fontdict=font_dict,
          loc='left',
          y=-0.35)

plt.subplot(122)
plt.boxplot(time, patch_artist=True,
            boxprops={'color':'blue','facecolor':'white', 'hatch':'\\', 'linewidth': 2},
            capprops={'linewidth':2, 'color': 'blue'},
            whiskerprops={'linewidth':2, 'color': 'blue'},
            # showfliers=False,
            flierprops={'linewidth':9, 'color': 'white', 'markerfacecolor':'red', 'marker': 'o'})
# plt.errorbar(np.array(range(1,5)), time, yerr=time_std, fmt='o-', ecolor='r', color='b', elinewidth=3, capsize=5)
plt.tick_params(labelsize=22)
plt.ylabel('Processing time (s)')
plt.xlabel('Video segment length (s)')
plt.grid(axis='y', linestyle='-')
plt.title('(b) The processing time for different video\nsegment lengths',
          fontdict=font_dict,
          loc='left',
          y=-0.35)
plt.tight_layout()


plt.savefig('../log/line_box.png')
plt.savefig('../log/line_box.pdf', format='pdf', dpi=600)
plt.show()
