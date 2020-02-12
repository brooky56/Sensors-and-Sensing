import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import statistics

curr_work_dir = os.getcwd()
data_arr = []


# read dir for current task
def read_dir(task_dir, work_dir):
    if os.path.exists(os.path.join(work_dir, task_dir)):
        read_dataset(os.path.join(work_dir, task_dir))


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

    plt.hist(angle, bins=100)
    plt.title("Angels values distribution before")
    plt.show()
    return angle_mean, angle_sdev, angle


def regression(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)

    d_xy = np.sum(y * x) - np.size(x) * y_mean * x_mean
    d_xx = np.sum(x * x) - np.size(x) * x_mean * x_mean

    b1 = d_xy / d_xx
    b0 = y_mean - b1 * x_mean

    return (b0, b1)


def plt_regression(x, y, b):
    plt.scatter(x, y, color="m",
                marker="o", s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_confidence_interval(angle_new):
    mean = np.mean(angle_new)
    ci_95 = 1.96 * stats.sem(angle_new)
    min = mean - ci_95
    max = mean + ci_95

    plt.figure(figsize=(15, 7))
    plt.plot(mean, "bo")
    plt.plot(min, "ro")
    plt.plot(max, "ro")
    plt.show()


def mean_confidence_interval(data, confidence=0.95):
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return m, m - h, m + h


if __name__ == '__main__':
    read_dir("case3.txt", curr_work_dir)
    mean, std, angle_arr = plot_before()

    filtered_angle = []
    for i in range(len(data_arr)):
        if (data_arr[i][1] > mean - std) and (data_arr[i][1] < mean + std):
            filtered_angle.append(np.array([data_arr[i][0], data_arr[i][1]]))

    time = np.array([filtered_angle[i][0] for i in range(len(filtered_angle))])
    angle_new = np.array([filtered_angle[i][1] for i in range(len(filtered_angle))])

    print("Min element: {0}".format(angle_new.min()))
    print("Max element: {0}".format(angle_new.max()))

    plt.plot(time, angle_new)
    plt.title("Angels(t) values after")
    plt.xlabel("Time")
    plt.ylabel("Angel")
    plt.show()

    b = regression(time, angle_new)
    print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plt_regression(time, angle_new, b)

    plot_confidence_interval(angle_new)

    print(mean_confidence_interval(angle_new))
    m, low, upper = mean_confidence_interval(angle_new)
    plt.hist(angle_new, bins=100)
    plt.plot([mean, mean], [0, 1300])
    plt.plot([low, low], [0, 1300])
    plt.plot([upper, upper], [0, 1300])
    plt.show()
