import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os

# logPath = '../log_w4/CL(sal+fix)/'
# logPath = '../log_w4/CL(frame+sal+fix)/'
# logPath = '../log_w8/CL(frame+sal+fix)/'
# logPath = '../log_w12/CL(frame+sal+fix)/'
logPath = '../log_w16/CL(frame+sal+fix)/'

nameList = ['Alien', 'Conan1', 'Cooking', 'Skiing']

num = 24
accuracy, F1 = [], []
beta = 1

for idx, name in enumerate(nameList):
    accuracyList, F1List = [], []
    for i in range(1, num+1):
        with open(logPath + f"{name}/{i}.csv", 'r') as csvfile:
            accList = []
            reader = csv.reader(csvfile)
            try:
                next(reader)
            except StopIteration:
                print(f'{name}, {i}.csv StopIteration')
                continue
            recall, precision = None, None
            for row in reader:
                if len(row) == 2:
                    if row[0] == "AverageRecall":
                        recall = float(row[1])
                    elif row[0] == "AveragePrecision":
                        precision = float(row[1])
                    else:
                        pass
                else:
                    acc = int(row[5]) / (int(row[3]) + int(row[4]) - int(row[5]))
                    accList.append(acc)
            accuracyList.append(np.mean(accList))
            F1_score = (1+beta*beta) * recall * precision / (beta*beta*precision + recall)
            F1List.append(F1_score)

    accuracy.append(np.mean(accuracyList))
    F1.append(np.mean(F1List))

x = np.arange(len(nameList))
width = 0.2
# plt.rcParams['font.size'] = 20
# fig, ax = plt.subplots(figsize=(12, 8))
fig, ax = plt.subplots()
n = 0

plt.bar(x + width * n, accuracy, width=width, label='Accuracy')
plt.bar(x + width * (n+1), F1, width=width, label=f'F{beta}-score')

plt.grid(linestyle='--')
plt.xticks(x + width / 2, nameList)
# plt.ylim(0, 1)
ax.set_ylabel('Metrics')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

plt.legend(loc='center', ncol=2, bbox_to_anchor=[0.5, 1.05])
# plt.legend(loc='center', bbox_to_anchor=[0.91, 1.12])
plt.tight_layout()
plt.savefig(f'{logPath}/acc_F{beta}.png')
plt.show()

fileName = f'{logPath}/acc_F{beta}.csv'
with open(fileName, 'w', newline='') as f:
    logWriter = csv.writer(f, dialect='excel')
    logWriter.writerow(['AverageMetric']+nameList)
    logWriter.writerows([
        ['Accuracy'] + accuracy,
        [f'F{beta}'] + F1,
    ])

