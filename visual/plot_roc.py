import csv
import os
import matplotlib.pyplot as plt

dir_path = "../log/Skiing/thres/"
fileList = os.listdir(dir_path)

print(len(fileList))

TPR_list = []
FPR_list = []
for file in fileList:
    with open(dir_path + file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        rows = [row for row in reader]
        if rows[-2][1] == 'nan':
            TPR = 1
        else:
            TPR = float(rows[-2][1])

        if rows[-1][1] == '0.0':
            FPR = 1
        else:
            FPR = float(rows[-1][1])
        TPR_list.append(TPR)
        FPR_list.append(FPR)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(FPR_list, TPR_list)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()