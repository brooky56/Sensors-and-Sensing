import cv2
import numpy as np
import imutils

src = "src"
results = "Results"


def difference(cup, empty):
    # get the difference between full and empty box
    diff = cup - empty
    cv2.imwrite("diff.jpg", diff)

    # inverse thresholding to change every pixel above 190
    # to black that means without the cup
    _, diff_th = cv2.threshold(diff, 190, 255, 1)
    cv2.imwrite("{0}diff_th.jpg".format(results), diff_th)

    return diff, diff_th


def mask(diff, diff_th):
    # combine the difference image and the inverse threshold
    # will give us just the cup
    cup = cv2.bitwise_and(diff, diff_th, None)
    cv2.imwrite("{0}\cup_only.jpg".format("Results"), cup)

    # threshold to get the mask instead of gray pixels
    _, cup = cv2.threshold(cup, 100, 255, 0)

    # dilate to account for the blurring in the beginning
    kernel = np.ones((15, 15), np.uint8)
    cup = cv2.dilate(cup, kernel, iterations=1)

    cv2.imwrite("{0}\cup.jpg".format(results), cup)
    return cup


if __name__ == '__main__':
    # load the images

    empty_shelf = cv2.imread("{0}\empty.jpg".format(src))
    cup = cv2.imread("{0}\cup_full.jpg".format(src))

    # save color copy for visualization
    cup_copy = cup.copy()

    # convert to grayscale
    empty_shelf_g = cv2.cvtColor(empty_shelf, cv2.COLOR_BGR2GRAY)
    cup_g = cv2.cvtColor(cup, cv2.COLOR_BGR2GRAY)

    empty_shelf_g = cv2.GaussianBlur(empty_shelf_g, (41, 41), 0)
    cup_g = cv2.GaussianBlur(cup_g, (41, 41), 0)

    diff, diff_threshold = difference(cup_g, empty_shelf_g)
    cup = mask(diff, diff_threshold)

    # find contours, sort and draw the biggest one
    contours, hierarchy = cv2.findContours(cup, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    cv2.drawContours(cup_copy, [contours[0]], -1, (0, 255, 0), 3)

    cv2.imwrite("{0}\Detected_cup.jpg".format(results), cup_copy)
