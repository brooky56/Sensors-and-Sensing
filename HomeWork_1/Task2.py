import numpy as np
import os
import matplotlib.pyplot as plt
import random
from mpl_toolkits import mplot3d

data_arr = []
curr_work_dir = os.getcwd()


# read dir for current task
def read_dir(task_dir, work_dir):
    if os.path.exists(os.path.join(work_dir, task_dir)):
        read_dataset(os.path.join(work_dir, task_dir))


# transformation matrix to 4x4
def transform(arr):
    # make arr filled with 1 [n x 4] and put there our points
    arr_t = np.ones((len(arr), 4))
    arr_t[:, :3] = arr
    return arr_t


# calculate plane model
def estimate(arr):
    arr_t = transform(arr[:3])
    return np.linalg.svd(arr_t)[-1][-1, :]


# main part for algorithm
def ransac(data, sample_set_amount, max_inliers, iterations, threshold):
    inlier_in = 0
    fitted_model = None

    data = list(data)
    for i in range(iterations):
        sample_set = random.sample(data, int(sample_set_amount))
        model = estimate(sample_set)
        inlier_count = 0
        for j in range(len(data)):
            if inlier(model, data[j], threshold):
                inlier_count += 1

        print("Sample set chosen: {0}".format(sample_set))
        print("Estimate model: {0}".format(model))
        print("Inliers in: {0}".format(inlier_count))

        if inlier_count > inlier_in:
            inlier_in = inlier_count
            fitted_model = model
            if inlier_count > max_inliers:
                break
    return fitted_model


def read_dataset(path):
    file = open(path)
    data = file.readline()
    while data:
        x, y, z = data.split(",")
        data_arr.append(np.array([float(x), float(y), float(z)]))
        data = file.readline()
    file.close()


def plot_dataset():
    _x = np.array([data_arr[i][0] for i in range(len(data_arr))])
    _y = np.array([data_arr[i][1] for i in range(len(data_arr))])
    _z = np.array([data_arr[i][2] for i in range(len(data_arr))])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(_x, _y, _z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


# check that point is in estimated limits for inlier
def inlier(coeffs, data, threshold):
    return np.abs(coeffs.dot(transform([data]).T)) < threshold


def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[:8, :8]
    return xx, yy, (-d - a * xx - b * yy) / c


if __name__ == '__main__':
    read_dir("data_set_8_.txt", curr_work_dir)
    plot_dataset()
    _x = np.array([data_arr[i][0] for i in range(len(data_arr))])
    _y = np.array([data_arr[i][1] for i in range(len(data_arr))])
    _z = np.array([data_arr[i][2] for i in range(len(data_arr))])

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)

    n = 2
    max_iterations = 100
    max_inliers = 100
    threshold = 0.01

    ax.scatter3D(_x, _y, _z)

    model = ransac(data_arr, n, max_inliers, max_iterations, threshold)

    # get model parameters
    a, b, c, d = model

    x, y, z = plot_plane(a, b, c, d)
    ax.scatter3D(_x, _y, _z)
    ax.plot_surface(x, y, z, color='r')
    plt.show()
