import numpy as np
import matplotlib.pyplot as plt
import csv

logPath = '../log/metrics.csv'

timeList = [1, 2, 3, 4]
accList, recallList, precisionList, F1List = [], [], [], []

with open(logPath, 'r') as csvfile:
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

fig, ax = plt.subplots(figsize=(8, 6))
# fig, ax = plt.subplots()
plt.rcParams['font.size'] = 16

ax.xaxis.set_major_locator(plt.MultipleLocator(1))

plt.plot(timeList, accList, marker='^', label='accuracy', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, recallList, marker='s', label='recall', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, precisionList, marker='d', label='precision', linestyle='dashed', linewidth=3.0, markersize=10)
plt.plot(timeList, F1List, marker='o', label='F1-score', linestyle='dashed', linewidth=3.0, markersize=10)
plt.xlabel('Prediction window (s)', fontsize=18)
plt.ylabel('Average metric', fontsize=18)
plt.tick_params(labelsize=15)
plt.legend(ncol=2)
plt.grid(linestyle='--')
# plt.tight_layout()
plt.savefig('../log/metric_avg.png')
plt.savefig('../log/metric_avg.pdf', format='pdf', dpi=600)
# plt.savefig('../log/metric.svg', format='svg')
plt.show()