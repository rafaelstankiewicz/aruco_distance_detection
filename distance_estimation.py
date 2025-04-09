import cv2 as cv
import numpy as np
from cv2 import aruco
import threading
from kalman import init_kalman, apply_kalman
from tcp_server import MarkerPoseServer

# To do: 
# 1. Implement Kalman filter for both orientation and detection
# 2. Walk through code line-by-line, review, clean up, make more object-oriented, commit to Clinic repo
# 3. Calibrate MBARI cameras, confirm unit of measurment
# 4. Try adaptive thresholding-- cv2.adaptiveThreshold
# 5. Verify marker size-- 3 in?
# 6. Debug rotation and translation vectors

# Establish TCP connection
pose_server = MarkerPoseServer()
server_thread = threading.Thread(target=pose_server.wait_for_connection, daemon=True)
server_thread.start()

calib_data_path = r"MultiMatrix.npz" # TODO: replace with MBARI camera calibration
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
MARKER_SIZE = 2.3  # Units = cm?

marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
param_markers =  cv.aruco.DetectorParameters()
param_markers.polygonalApproxAccuracyRate = 0.03
param_markers.maxErroneousBitsInBorderRate = 0.5
detector = cv.aruco.ArucoDetector(marker_dict, param_markers)

cap = cv.VideoCapture("1.MOV")  # Change to ROV video stream

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv.CAP_PROP_FPS))

output_path = "/mnt/c/Users/rstan/Downloads/kalman_filter_test_footage.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

kalman_filters = {}  # Per marker

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame,(490,852))

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray_frame)

    if ids is not None:  # If corners?
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, ids.size)
        # if marker_IDs.size > 1 or (marker_IDs.size == 1 and marker_IDs[0][0]!=999):
        #     print(marker_IDs)

        positions = []  # Store each marker's (x, z) position and ID

        for marker_id, corners, i in zip(ids.flatten(), corners, total_markers):
            x, y, z = tVec[i][0]  # Extract raw x, y, z coordinates

            if marker_id not in kalman_filters:
                kalman_filters[marker_id] = init_kalman()

            # Apply Kalman filter
            measurement = np.array([[x], [z]], dtype=np.float32)
            x_kalman, z_kalman = apply_kalman(kalman_filters[marker_id], measurement)

            # Send pose to Unity
            pose_server.send_pose(marker_id, tVec[i][0], rVec[i][0])

            positions.append((marker_id, x_kalman, z_kalman))  # Store smoothed positions

            # Draw marker borders and annotations
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            # top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            # bottom_left = corners[3].ravel()
            
            # Calculate distance use smoothed coordinates
            distance = np.sqrt(x_kalman**2 + z_kalman**2)
            
            # Display local coordinate frame of each marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {marker_id} Dist: {np.round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

            cv.putText(
                frame,
                f"x:{np.round(x_kalman,1)} z:{np.round(z_kalman,1)}",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        # # Check if markers are aligned in (x, z) position
        # freeze_frame = False
        # for i in range(len(positions)):
        #     for j in range(i+1, len(positions)):
        #         id1, x1, z1 = positions[i]
        #         id2, x2, z2 = positions[j]

        #         # Alignment threshold
        #         if abs(x1-x2) < 2 and abs(z1-z2) < 2:
        #             cv.putText(
        #                 frame,
        #                 f"Markers {id1} and {id2} are aligned",
        #                 (50, 50),  # Position to display text from top left corner of frame
        #                 cv.FONT_HERSHEY_SIMPLEX,
        #                 1,
        #                 (0, 255, 0),
        #                 2,
        #                 cv.LINE_AA,
        #             )

        #             cv.imshow("frame", frame)
        #             cv.waitKey(5000)  # Pauses video for 5 sec in video player 

        #             freeze_frame = True  # Marker alignment bool
        #             prev_frame = frame.copy()
        #             break

        #     # print(ids, "  ", corners)

        # # Freeze output video for 5 sec when markers are aligned
        # if freeze_frame and prev_frame is not None:
        #     for _ in range(int(fps * 5)):
        #         out.write(prev_frame)

    out.write(frame)
    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord('p'):
        cv.waitKey(-1) # Wait until any key is pressed

cap.release()
out.release()
cv.destroyAllWindows()
pose_server.close()
