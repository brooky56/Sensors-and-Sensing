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


if __name__ == '__main__':
    read_dir("case3.txt", curr_work_dir)
    plot_before()

    currrent = []
    filtered = []
    K_coef = []

    xk_time = data_arr[0][0]
    yk_angle = data_arr[0][1]

    # model mse
    model_mse = 0.003
    # sensor mse
    sensor_mse = 0.2

    i = 1
    for i in range(len(data_arr)):
        dx = data_arr[i][0] - xk_time
        xk_time = data_arr[i][0]
        u = 3 * cos(xk_time) * dx
        # calculating formulas to gain K parameter
        K = (sensor_mse ** 2 + model_mse ** 2) / (sensor_mse ** 2 + model_mse ** 2 + model_mse ** 2)

        mse = sensor_mse ** 2 * (sensor_mse ** 2 + model_mse ** 2) / (
                sensor_mse ** 2 + sensor_mse ** 2 + model_mse ** 2)

        yk_angle = K * data_arr[i][1] + (1 - K) * (yk_angle + u)

        filtered.append(yk_angle)
        currrent.append(data_arr[i][1])

    plt.plot(currrent, c='g')
    plt.plot(filtered, c='b')
    plt.show()
