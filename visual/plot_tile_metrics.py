import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os

# logPath = '../log/CL(frame+sal+fix)/'
# logPath = '../log/CL(sal+fix)/'
# logPath = '../log/CL(sal+fix)-5/'
logPath = '../log_w4/CL(frame+sal+fix)/'
# logPath = '../log_w8/CL(frame+sal+fix)/'
# logPath = '../log_w8/CL(frame+fix)/'
# logPath = '../log_w12/CL(frame+sal+fix)/'
# logPath = '../log_w16/CL(frame+sal+fix)/'

nameList = ['Alien', 'Conan1', 'Cooking', 'Skiing']

num = 24
accuracy, recall, precision = [], [], []
time = []

for idx, name in enumerate(nameList):
    accuracyList, recallList, precisionList = [], [], []
    timeList = []
    for i in range(1, num+1):
        with open(logPath + f"{name}/{i}.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

            try:
                timeList.append(round(float(rows[-8][1]), 4))
                accuracyList.append(round(float(rows[-3][1]), 4))
                recallList.append(round(float(rows[-2][1]), 4))
                precisionList.append(round(float(rows[-1][1]), 4))
            except IndexError:
                print("IndexError:", idx)

    time.append(np.mean(timeList))
    accuracy.append(np.mean(accuracyList))
    recall.append(np.mean(recallList))
    precision.append(np.mean(precisionList))

x = np.arange(len(nameList))
width = 0.15
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(12, 8))
n = 0

plt.bar(x + width * n, accuracy, width=width, label='AvgAccuracy')
plt.bar(x + width * (n+1), recall, width=width, label='AvgRecall')
plt.bar(x + width * (n+2), precision, width=width, label='AvgPrecision')

plt.grid(linestyle='--')
plt.xticks(x + width, nameList, fontsize=20)
plt.ylim(0, 1)
ax.set_ylabel('AverageMetrics')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

plt.legend(loc='center', ncol=3, bbox_to_anchor=[0.5, 1.05])
# plt.legend(loc='center', bbox_to_anchor=[0.91, 1.12])
# plt.title("Attention offline mode")
plt.tight_layout()
plt.savefig(f'{logPath}/avg.png')
# plt.savefig(f'{logPath}/avg.svg', format='svg', dpi=150)
plt.show()

fileName = f'{logPath}/avg.csv'
with open(fileName, 'w', newline='') as f:
    logWriter = csv.writer(f, dialect='excel')
    logWriter.writerow(['AverageMetric']+nameList)
    logWriter.writerows([
        ['Accuracy'] + accuracy,
        ['Recall'] + recall,
        ['Precision'] + precision,
        ['Time'] + time,
    ])

