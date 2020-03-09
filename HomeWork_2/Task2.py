import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# checkerboard Dimensions
cb_row = 4
cb_col = 7

obj_p = np.zeros((cb_row * cb_col, 3), np.float32)
obj_p[:, :2] = np.mgrid[0:cb_col, 0:cb_row].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.

images = glob.glob('img/*.png')


# First we estimate our calibrating board, finding corners, for all list of photos
def chess_board_estimation():
    for fname in images:
        img = cv2.imread(fname)
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cb_col, cb_row), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            obj_points.append(obj_p)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 4), corners2, ret)
            cv2.imshow(img)

    return gray


def camera_calibration(gray):
    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # print(np.asarray(objpoints).tolist())
    imgp = []
    a = []
    for i in range(len(img_points)):
        for j in range(len(img_points[0])):
            x = img_points[i][j][0][0]
            y = img_points[i][j][0][1]
            a.append([x, y])
        imgp.append(np.array(a))
        a = []

    data = {'Intrinsic camera params': np.asarray(mtx).tolist(), 'Distortion coefficients': np.asarray(dist).tolist()}

    cv2.solvePnP(np.asarray(obj_points[0], dtype=np.float32), np.asarray(imgp[0], dtype=np.float32),
                 np.asarray(mtx, dtype=np.float32), np.asarray(dist, dtype=np.float32))
    print(data)

    # Return Intrinsic and Extrinsic camera parameters for further reconstructing
    return mtx, rvecs, tvecs


# When we know all parms we can estimate coordinates in 3D knowing coordinates in camera plane
# All we should do: lambada * [x_c y_c 1] = K * [R|T] * [X_w Y_w Z_w 1]
def estimate_object(mtx, R, T):
    filename = 'object.png'
    im = cv2.imread(filename)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin = cv2.threshold(gray, 120, 255, 1)  # inverted threshold (light obj on dark bg)

    bin = cv2.dilate(bin, None)  # fill some holes
    bin = cv2.dilate(bin, None)
    bin = cv2.erode(bin, None)  # dilate made our shape larger, revert that
    bin = cv2.erode(bin, None)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rc = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rc)
    for p in box:
        pt = (p[0], p[1])
        print(pt)
        cv2.circle(im, pt, 5, (200, 0, 0), 2)
    cv2.imshow(im)

    v = []
    world_cor = []
    for i in range(len(box)):
        v.append([box[i][0], box[i][1], 1])
        v = np.array(v)
        # print(v)
        b = np.linalg.inv(mtx * R * T.transpose())
        world_cor.append(np.dot(b, v.transpose()))
        v = []

    print("Coordinates in world axis: {0}".format(world_cor))


if __name__ == '__main__':
    gray = chess_board_estimation()
    K, R, T = camera_calibration(gray)
    estimate_object(K, R, T)
