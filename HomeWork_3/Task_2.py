# import modules
import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
from numpy import sin, cos, pi
import matplotlib.pyplot as plt


# import data from CSV
def read_dataset(name):
    dataset = pd.read_csv('src\{0}'.format(name))
    return dataset


# Convert orientations to rad
def convert_rad(dataset):
    cols_angles = ['ORIENTATION X (pitch °)', 'ORIENTATION Y (roll °)', 'ORIENTATION Z (azimuth °)']
    for axis in cols_angles:
        dataset[axis] = dataset[axis] * pi / 180


# Rotations matrices
def R_x(x):
    return np.array([[1, 0, 0],
                     [0, cos(-x), -sin(-x)],
                     [0, sin(-x), cos(-x)]])


def R_y(y):
    return np.array([[cos(-y), 0, -sin(-y)],
                     [0, 1, 0],
                     [sin(-y), 0, cos(-y)]])


def R_z(z):
    return np.array([[cos(-z), -sin(-z), 0],
                     [sin(-z), cos(-z), 0],
                     [0, 0, 1]])


def transform_accel_Earth_frame(dataset, accel, roll, pitch, yaw, grav, line):
    earth_accels = np.empty(accel.shape)
    earth_gravity = np.empty(accel.shape)
    earth_linear = np.empty(accel.shape)
    # Transformations mobile frame to Earth frame
    for i in range(dataset.shape[0]):
        earth_accels[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ accel[:, i]
        earth_gravity[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ grav[:, i]
        earth_linear[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ line[:, i]

    # Create new columns in dataframe for earth frame accelerations
    dataset['EARTH ACCELERATION X'] = earth_accels[0, :]
    dataset['EARTH ACCELERATION Y'] = earth_accels[1, :]
    dataset['EARTH ACCELERATION Z'] = earth_accels[2, :]
    dataset['EARTH GRAVITY X'] = earth_gravity[0, :]
    dataset['EARTH GRAVITY Y'] = earth_gravity[1, :]
    dataset['EARTH GRAVITY Z'] = earth_gravity[2, :]
    dataset['EARTH LINEAR ACCELERATION X'] = earth_linear[0, :]
    dataset['EARTH LINEAR ACCELERATION Y'] = earth_linear[1, :]
    dataset['EARTH LINEAR ACCELERATION Z'] = earth_linear[2, :]

    # Plot new accelerations
    cols_earth = ['EARTH ACCELERATION X', 'EARTH ACCELERATION Y',
                  'EARTH ACCELERATION Z', 'EARTH GRAVITY X', 'EARTH GRAVITY Y', 'EARTH GRAVITY Z',
                  'EARTH LINEAR ACCELERATION X', 'EARTH LINEAR ACCELERATION Y', 'EARTH LINEAR ACCELERATION Z']
    cols_body = ['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)',
                 'ACCELEROMETER Z (m/s²)', 'GRAVITY X (m/s²)', 'GRAVITY Y (m/s²)', 'GRAVITY Z (m/s²)',
                 'LINEAR ACCELERATION X (m/s²)',
                 'LINEAR ACCELERATION Y (m/s²)', 'LINEAR ACCELERATION Z (m/s²)']

    return dataset, cols_earth, cols_body


def mobile_position(dataset, dt):
    # Double integrate accelerations to find positions
    x = cumtrapz(cumtrapz(dataset['EARTH LINEAR ACCELERATION X'], dx=dt), dx=dt)
    y = cumtrapz(cumtrapz(dataset['EARTH LINEAR ACCELERATION Y'], dx=dt), dx=dt)
    z = cumtrapz(cumtrapz(dataset['EARTH LINEAR ACCELERATION Z'], dx=dt), dx=dt)
    # Plot 3D Trajectory
    fig3, ax = plt.subplots()
    fig3.suptitle('3D Trajectory of phone', fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, c='green', lw=5, label='phone trajectory')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    plt.savefig('{}\Trajectory'.format('Results'))


def remove_noise(dataset, dt):
    freq = np.fft.rfftfreq(dataset['EARTH LINEAR ACCELERATION X'].size, d=dt)
    # Compute the Fast Fourier Transform (FFT) of acceleration signals
    fft_x = np.fft.rfft(dataset['EARTH LINEAR ACCELERATION X'])
    fft_y = np.fft.rfft(dataset['EARTH LINEAR ACCELERATION Y'])
    fft_z = np.fft.rfft(dataset['EARTH LINEAR ACCELERATION Z'])

    # Attenuate noise in X,Y below 1Hz by 0.2
    atten_x_fft = np.where(freq < 15, fft_x * 0.1, fft_x)
    atten_y_fft = np.where(freq < 15, fft_y * 0.1, fft_y)
    atten_z_fft = np.where((freq > 2) & (freq < 15), fft_z * 0.1, fft_z)
    # Compute inverse of discrete Fourier Transform and save to dataframe
    dataset['x_ifft'] = np.fft.irfft(atten_x_fft, n=dataset.shape[0])
    dataset['y_ifft'] = np.fft.irfft(atten_y_fft, n=dataset.shape[0])
    dataset['z_ifft'] = np.fft.irfft(atten_z_fft, n=dataset.shape[0])
    # Double integrate accelerations to calculate coordinate positions
    x = cumtrapz(cumtrapz(dataset['x_ifft'], dx=dt), dx=dt)
    y = cumtrapz(cumtrapz(dataset['y_ifft'], dx=dt), dx=dt)
    z = cumtrapz(cumtrapz(dataset['z_ifft'], dx=dt), dx=dt)
    return dataset, x, y, z


def KL_GPS_altitude(dataset, dt):
    # State Transition Matrix
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # Control Matrix
    B = np.array([[0.5 * dt ** 2],
                  [dt]])

    # Control vector
    u = np.array([[0.0]])

    # Process Noise Matrix
    max_position_change = 0.5 * 0.5 * dt ** 2  # assume max accel = 0.5
    max_velocity_change = 0.5 * dt  # assume max accel = 0.5
    Q = np.array([[max_position_change, 0],
                  [0, max_velocity_change]])

    # Measurement Matrix
    H = np.array([[1, 0]])  # Only able to measure position

    # Measurement noise covariance matrix
    sbarometer = 1.0  # accurate to +- 1m
    R = np.array([[sbarometer ** 2]])

    # Measurement
    z_m = np.array([[0.0]])

    # Initial System State Matrix (pos_z = 0, vel_z = 0 at t = 0)
    X = np.array([[0.0],
                  [0.0]])

    # Initial Process Covariance Matrix
    spos = 0.0  # No uncertainty in initial state
    svel = 0.0
    P = np.array([[spos ** 2, svel * spos],
                  [spos * svel, svel ** 2]])
    # 2x2 Identity Matrix
    I = np.eye(2)

    X_pos = []
    X_vel = []
    P_pos = []
    P_vel = []
    K_pos = []
    K_vel = []

    for i in range(dataset.shape[0]):

        # Pull in z acceleration control input
        u[0][0] = dataset['z_ifft'][i]

        # Predict the next state
        X = A @ X + B @ u
        P = A @ P @ A.T + Q

        # Altitude measurement every 15 accelerometer updates (dt=0.01s vs dt_barometer = 0.015s)
        if i % 5 == 0:
            # Pull in altitude measurement
            z_m[0][0] = dataset['LOCATION Altitude ( m)'][i] * 0.3048

            # Update the next state
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)  # Kalman Gain
            X = X + K @ (z_m - H @ X)  # Updated State
            P = (I - K @ H) @ P  # Updated Covariance

        # --- Store system states variables, Kalman Gain, and covariances ---
        X_pos.append(X[0][0])
        X_vel.append(X[1][0])
        P_pos.append(P[0][0])
        P_vel.append(P[1][1])
        K_pos.append(K[0][0])
        K_vel.append(K[1][0])

    return X_pos, X_vel, K_pos, K_vel
