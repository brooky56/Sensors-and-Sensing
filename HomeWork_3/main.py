from HomeWork_3.Task_2 import *

if __name__ == '__main__':
    dataset = read_dataset("Sensor_record_2.csv")
    src = "Results"
    # All sensors outputs
    dataset.plot(subplots=True, sharex=True, layout=(6, 6))
    plt.savefig('{}\Every_Sensor_datastream_of_mobile_phone'.format(src))

    # Main mobile sensors data
    acceleration = np.array([dataset['ACCELEROMETER X (m/s²)'],
                             dataset['ACCELEROMETER Y (m/s²)'],
                             dataset['ACCELEROMETER Z (m/s²)']])
    gravity = np.array([dataset['GRAVITY X (m/s²)'],
                        dataset['GRAVITY Y (m/s²)'],
                        dataset['GRAVITY Z (m/s²)']])
    linear_acceleration = np.array([dataset['LINEAR ACCELERATION X (m/s²)'],
                                    dataset['LINEAR ACCELERATION Y (m/s²)'],
                                    dataset['LINEAR ACCELERATION Z (m/s²)']])

    # Euler angles for rotation matrices
    pitch = dataset['ORIENTATION X (pitch °)']
    roll = dataset['ORIENTATION Y (roll °)']
    yaw = dataset['ORIENTATION Z (azimuth °)']

    dataset, cols_earth, cols_body = transform_accel_Earth_frame(dataset, acceleration, roll, pitch, yaw, gravity,
                                                                 linear_acceleration)

    axes = dataset.plot(y=cols_body, subplots=True, sharex=True, layout=(3, 3), style='k', alpha=0.5,
                        title=cols_body)
    dataset.plot(y=cols_earth, subplots=True, layout=(3, 3), ax=axes,
                 sharex=True, style='g',
                 title='Body Frame to Earth Frame Accelerations')
    plt.savefig('{}\Body_Frame_to_Earth_Frame_Accelerations'.format(src))

    sampling_rate = 0.01

    mobile_position(dataset, sampling_rate)

    dataset, x, y, z = remove_noise(dataset, sampling_rate)

    Z_pos, Z_vel, K_pos, K_vel = KL_GPS_altitude(dataset, sampling_rate)

    # Kalman Gain
    fig2, axs2 = plt.subplots()
    axs2.set_title('Kalman Gain for state variables',
                   fontsize=20)
    axs2.set_xlabel('Time (ms)', fontsize=15)
    axs2.plot(K_pos, label='Kalman Gain for Position')
    axs2.plot(K_vel, label='Kalman Gain for Velocity')
    axs2.set_ylabel('Kalman Gain', fontsize=25)
    axs2.legend(fontsize=12)
    plt.savefig('{}\Kalman_Gain'.format(src))

    # Plot new trajectory on 3D plot
    fig3, a3 = plt.subplots()
    fig3.suptitle('Kalman Filtered 3D phone trajectory', fontsize=20)
    ax3 = plt.axes(projection='3d')
    ax3.plot3D(x, y, Z_pos[:-2], c='green', lw=5, label='Kalman Filtered phone trajectory')
    ax3.set_xlabel('X position (m)')
    ax3.set_ylabel('Y position (m)')
    ax3.set_zlabel('Z position (m)')
    plt.savefig('{}\Filtered Trajectory'.format(src))
    plt.show()
