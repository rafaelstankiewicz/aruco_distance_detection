import cv2 as cv
from cv2 import aruco
import numpy as np

# To do: 
# 1. Implement Kalman filter and marker alignment and axil display improves
# 2. Walk through code line-by-line for understanding
# 3. Clean up code, make more object-oriented, commit to Clinic repo
# 4. Calibrate iPhone camera?

calib_data_path = r"MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
MARKER_SIZE = 2.3  # in cm

marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)
param_markers =  cv.aruco.DetectorParameters()
param_markers.polygonalApproxAccuracyRate = 0.03
param_markers.maxErroneousBitsInBorderRate = 0.5
detector = cv.aruco.ArucoDetector(marker_dict, param_markers)

cap = cv.VideoCapture("./dewarpedVideos/whiteBorderDemo3.mp4")

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

output_path = "output.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Because the sampler footage is at such a high resolution.. scaling it down for display purposes
def rescaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # R, G, B = cv.split(frame)
    # output_R = cv.equalizeHist(R)
    # output_G = cv.equalizeHist(G)
    # output_B = cv.equalizeHist(B)

    # frame = cv.merge((output_R, output_G, output_B))

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray_frame = cv.equalizeHist(gray_frame)


    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        # if marker_IDs.size > 1 or (marker_IDs.size == 1 and marker_IDs[0][0]!=999):
        #     print(marker_IDs)

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
        freeze_frame = False
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                id1, x1, z1 = positions[i]
                id2, x2, z2 = positions[j]

                # Alignment threshold
                if abs(x1-x2) < 2 and abs(z1-z2) < 2:
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

                    cv.imshow("frame", frame)
                    cv.waitKey(5000)  # Pauses video for 5 sec in video player 

                    freeze_frame = True  # Marker alignment bool
                    prev_frame = frame.copy()
                    break

            # print(ids, "  ", corners)

        # Freeze output video for 5 sec when markers are aligned
        if freeze_frame and prev_frame is not None:
            for _ in range(int(fps * 5)):
                out.write(prev_frame)

    out.write(frame)

    frame = rescaleFrame(frame, percent=60)
    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord('p'):
        cv.waitKey(-1) # Wait until any key is pressed

cap.release()
out.release()
cv.destroyAllWindows()
