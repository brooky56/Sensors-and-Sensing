import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
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


# plotting elements before removing outliers
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


# Simple regression minimize total residual error
def regression(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)

    # s_xy is the sum of cross-deviations of y and x
    s_xy = np.sum(y * x) - np.size(x) * y_mean * x_mean
    # s_xx is the sum of cross-deviations of x
    s_xx = np.sum(x * x) - np.size(x) * x_mean * x_mean

    b1 = s_xy / s_xx
    b0 = y_mean - b1 * x_mean

    return b0, b1


def plt_regression(x, y, b):
    plt.scatter(x, y, color="m", marker="o")
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def plot_confidence_interval_points(angle_new):
    mean = np.mean(angle_new)
    ci_95 = 1.96 * stats.sem(angle_new)
    min = mean - ci_95
    max = mean + ci_95

    plt.plot(mean, "bo")
    plt.plot(min, "ro")
    plt.plot(max, "ro")
    plt.title("Confidence interval points")
    plt.show()


# find confidence interval
def plot_ci(data, confidence=0.95):
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)

    plt.hist(angle_new, bins=100)
    plt.plot([mean, mean], [0, 1300])
    plt.plot([m - h, m - h], [0, 1300])
    plt.plot([m + h, m + h], [0, 1300])
    plt.show()


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

    plt.hist(angle_new, bins=100)
    plt.title("Angels values distribution after")
    plt.show()

    b = regression(time, angle_new)
    print("Coeff estimated : b0 = {0}, b_1 = {1}".format(b[0], b[1]))

    # plotting regression line
    plt_regression(time, angle_new, b)

    # plotting bounds for ci
    plot_confidence_interval_points(angle_new)

    # plotting interval on histogram with distribution
    plot_ci(angle_new)
