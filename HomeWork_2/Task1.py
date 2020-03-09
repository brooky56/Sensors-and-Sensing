import os
import numpy as np
import matplotlib.pyplot as plt
from math import *
import statistics

# current dir
curr_work_dir = os.getcwd()
data_arr = []


# read dir for current task
def read_dir(task_dir, work_dir):
    if os.path.exists(os.path.join(work_dir, task_dir)):
        read_dataset(os.path.join(work_dir, task_dir))


# read dataset form file
def read_dataset(path):
    file = open(path)
    data = file.readline()
    while data:
        x, y = data.split(",")
        data_arr.append(np.array([float(x), float(y)]))
        data = file.readline()
    file.close()


def plot_before():
    t = np.array([data_arr[i][0] for i in range(len(data_arr))])
    angle = np.array([data_arr[i][1] for i in range(len(data_arr))])
    print("Amount of elements: {0}".format(len(angle)))

    angle_mean = statistics.mean(angle)
    angle_sdev = statistics.stdev(angle)

    print("Mean: {0}".format(angle_mean))
    print("Standart deviation: {0}".format(angle_sdev))

    plt.plot(t, angle)
    plt.title("Angles(t) values before")
    plt.xlabel("Time")
    plt.ylabel("Angel")
    plt.show()

    return angle_mean, angle_sdev, angle


std_e = 0.003
std_n = 0.2

E = std_n ** 2

signal = []
prediction = []
Ks = []

if __name__ == '__main__':
    read_dir("case3.txt", curr_work_dir)
    plot_before()

    x0 = data_arr[0][0]
    y0 = data_arr[0][1]

    i = 1
    for i in range(len(data_arr)):
        dx = data_arr[i][0] - x0
        x0 = data_arr[i][0]
        K = (E + std_e ** 2) / (E + std_n ** 2 + std_e ** 2)
        E = std_n ** 2 * (E + std_e ** 2) / (E + std_n ** 2 + std_e ** 2)

        y0 = K * data_arr[i][1] + (1 - K) * (y0 + 5 * dx)

        Ks.append(K)
        prediction.append(y0)
        signal.append(data_arr[i][1])

    plt.plot(signal, label='data', )
    plt.plot(prediction, label='filtered', c='black')
    plt.show()