import numpy as np
import matplotlib.pyplot as plt
import csv

logPath = '../log/metrics.csv'

timeList = [1, 2, 3, 4]
accList, recallList, precisionList, F1List = [], [], [], []

with open(logPath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    accList.append(float(rows[1][3]))
    accList.append(float(rows[5][3]))
    accList.append(float(rows[9][3]))
    accList.append(float(rows[13][3]))

    recallList.append(float(rows[2][3]))
    recallList.append(float(rows[6][3]))
    recallList.append(float(rows[10][3]))
    recallList.append(float(rows[14][3]))

    precisionList.append(float(rows[3][3]))
    precisionList.append(float(rows[7][3]))
    precisionList.append(float(rows[11][3]))
    precisionList.append(float(rows[15][3]))

    F1List.append(float(rows[4][3]))
    F1List.append(float(rows[8][3]))
    F1List.append(float(rows[12][3]))
    F1List.append(float(rows[16][3]))

fig, ax = plt.subplots(figsize=(8, 6))
# fig, ax = plt.subplots()
plt.rcParams['font.size'] = 16

ax.xaxis.set_major_locator(plt.MultipleLocator(1))

plt.plot(timeList, accList, marker='^', label='accuracy', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, recallList, marker='s', label='recall', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, precisionList, marker='d', label='precision', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, F1List, marker='o', label='F1-score', linestyle='dashed', linewidth=3.0, markersize=10)
plt.xlabel('Prediction window (s)', fontsize=18)
plt.ylabel('Metric', fontsize=18)
plt.tick_params(labelsize=15)
plt.legend(ncol=2)
plt.grid(linestyle='--')
# plt.tight_layout()
plt.savefig('../log/metric.png')
plt.savefig('../log/metric.pdf', format='pdf', dpi=600)
# plt.savefig('../log/metric.svg', format='svg')
plt.show()