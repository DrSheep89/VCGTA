import numpy as np
import matplotlib.pyplot as plt
import csv

logPath = '../log/F1_acc_models.csv'

accList = [[], [], [], []]
F1List = [[], [], [], []]

with open(logPath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

    nameList = rows[0][1:]

    accList[0] = [float(item) for item in rows[1][1:]]
    accList[1] = [float(item) for item in rows[2][1:]]
    accList[2] = [float(item) for item in rows[3][1:]]
    accList[3] = [float(item) for item in rows[4][1:]]

    F1List[0] = [float(item) for item in rows[5][1:]]
    F1List[1] = [float(item) for item in rows[6][1:]]
    F1List[2] = [float(item) for item in rows[7][1:]]
    F1List[3] = [float(item) for item in rows[8][1:]]


# TileMetrics
x = np.arange(len(nameList))
width = 0.2
plt.rcParams['font.size'] = 20
fig = plt.figure(figsize=(18, 7))
# fig = plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.bar(x, accList[0], hatch='x', width=width, label='ConvLSTM(fsu)')
plt.bar(x + width, accList[1], hatch='\\', width=width, label='ConvLSTM(su)')
plt.bar(x + width * 2, accList[2], hatch='/', width=width, label='LiveDeep')
plt.bar(x + width * 3, accList[3], hatch='*', width=width, label='LiveObj')
# plt.grid(linestyle='--')
plt.xticks(x + width * 3 / 2, nameList)
plt.ylabel('Accuracy')
plt.xlabel('Video Name')

# plt.xticks(x + width, nameList, fontsize=20)
# plt.ylim(0, 1)
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# plt.legend(loc='center', ncol=4, fontsize=14, bbox_to_anchor=[0.5, 1.05])
# plt.legend(loc='best')
# plt.tick_params(labelsize=18)

plt.subplot(122)
plt.bar(x, F1List[0], hatch='x', width=width, label='ConvLSTM(fsu)')
plt.bar(x + width, F1List[1], hatch='\\', width=width, label='ConvLSTM(su)')
plt.bar(x + width * 2, F1List[2], hatch='/', width=width, label='LiveDeep')
plt.bar(x + width * 3, F1List[3], hatch='*', width=width, label='LiveObj')
# plt.grid(linestyle='--')
plt.xticks(x + width * 3 / 2, nameList)
plt.ylabel('F1-score')
plt.xlabel('Video Name')

# plt.xticks(x + width, nameList, fontsize=20)
# plt.ylim(0, 1)
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# plt.legend(loc='center', ncol=4, fontsize=15, bbox_to_anchor=[0.5, 1.05])
# plt.legend(loc='best')
# plt.tick_params(labelsize=18)
lines, labels = fig.axes[-1].get_legend_handles_labels()
print(lines, labels)

fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=[0.5, 1])

# plt.tight_layout()
# plt.savefig('../log/F1_acc.png')
plt.savefig('../log/F1_acc.pdf', format='pdf', dpi=600)
# plt.savefig('../log/F1_acc.svg', format='svg', dpi=600, bbox_inches='tight')
plt.show()
