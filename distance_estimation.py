import cv2 as cv
from cv2 import aruco
import numpy as np

calib_data_path = r"MultiMatrix.npz"  # Holds calibration data?
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
MARKER_SIZE = 2.3  # in cm

marker_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)
param_markers = aruco.DetectorParameters_create()
cap = cv.VideoCapture("./videos/1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)

        # List to store each marker's (x, z) position and ID
        positions = []

        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            x, y, z = tVec[i][0]  # Extract x, y, z coordinates
            positions.append((ids[0], x, z))

            # Draw marker borders and annotations
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            
            # Calculate distance
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )
            
            # Display local coordinate frame of each marker
            point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            cv.putText(
                frame,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

            cv.putText(
                frame,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        # Check if any two markers are aligned in (x, z) position
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                id1, x1, z1 = positions[i]
                id2, x2, z2 = positions[j]

                # Alignment threshold
                if abs(x1-x2) < 0.1 and abs(z1-z2) < 0.1:
                    cv.putText(
                        frame,
                        f"Markers {id1} and {id2} are aligned",
                        (50, 50),  # Position to display text from top left corner of frame
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(5)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
