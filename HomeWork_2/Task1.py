import os
import numpy as np
import matplotlib.pyplot as plt
from math import *

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


if __name__ == '__main__':
    read_dir("case3.txt", curr_work_dir)
