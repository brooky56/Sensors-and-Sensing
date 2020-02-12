import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

from mpl_toolkits import mplot3d

data_arr = []
curr_work_dir = os.getcwd()


# read dir for current task
def read_dir(task_dir, work_dir):
    if os.path.exists(os.path.join(work_dir, task_dir)):
        read_dataset(os.path.join(work_dir, task_dir))


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        print(s)
        print('estimate:', m, )
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic


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


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


if __name__ == '__main__':
    read_dir("data_set_8_.txt", curr_work_dir)
    plot_dataset()
    _x = np.array([data_arr[i][0] for i in range(len(data_arr))])
    _y = np.array([data_arr[i][1] for i in range(len(data_arr))])
    _z = np.array([data_arr[i][2] for i in range(len(data_arr))])

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)


    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:10, :10]
        return xx, yy, (-d - a * xx - b * yy) / c


    n = 1000
    max_iterations = 1000
    goal_inliers = n * 0.3

    # test data
    # xyzs = np.random.random((n, 3)) * 10
    # xyzs[:50, 2:] = xyzs[:50, :1]

    ax.scatter3D(_x, _y, _z)

    # RANSAC
    m, b = run_ransac(data_arr, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color='g')

    plt.show()
