import cv2 as cv
import numpy as np

def init_kalman():
    kalman = cv.KalmanFilter(4, 2)

    kalman.transitionMatrix = np.array([  # State transition matrix, assuming constant velocity
        [1, 0, 1, 0], # x' = x + vx * dt
        [0, 1, 0, 1],  # z' = z +vz * dt
        [0, 0, 1, 0],  # vx' = vx
        [0, 0, 0, 1]   # vz' = vz
    ], dtype=np.float32)

    kalman.measurementMatrix = np.array([   # Measurement matrix
        [1, 0, 0, 0],  # x
        [0, 1, 0, 0]   # z
    ], dtype=np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 3e-2  # Process noise covariance
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  # Measurement noise covariance
    kalman.errorCov = np.eye(4, dtype=np.float32) * 1e-1  # State error covariance
    kalman.state = np.zeros(4, dtype=np.float32)  # Initial state estimate

    return kalman

def apply_kalman(kalman, measurement):
     """Applies Kalman filter to smooth marker position (x, z)."""
     kalman.correct(measurement)  # Correct with new measurement
     predicted = kalman.predict() # Predict next state
     return predicted[0], predicted[1]  # (x, z)