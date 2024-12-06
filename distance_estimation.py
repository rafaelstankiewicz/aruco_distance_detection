import cv2 as cv
from cv2 import aruco
import numpy as np
import overlay as ol

calib_data_path = r"MultiMatrix.npz"  # Holds calibration data
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

        

        


        



cap = cv.VideoCapture("5.mov")
print(cap.get(cv.CAP_PROP_FRAME_HEIGHT),cap.get(cv.CAP_PROP_FRAME_WIDTH))
cyl = ol.trackerOverlay(cap,999,size = (3,10),position = (0,10,-4),alpha = 0.5,line_thickness=2)
cyl.makeWireFrame(7,3)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
framesize = (480,852)
out = cv.VideoWriter('output2.mp4', fourcc, cap.get(cv.CAP_PROP_FPS),framesize)

line_count = 0
lyrics = "Piping hot! (popping like popcorn) It's all mine! (can you hear the bells ding-dong?) This is it! this is it! Whoo-oo-hoo! (guitar riff) If I could just see your heart, how small the world. It's not that I don't want to know. Walking and stopping have a meaning (I guess so!) What can i say, how many miles is liiiife? grab lots of snacks, and wear new shooo-oes! HOW FAR CAN I GO? I just want to feel like that every daaaaay! (drums) PIPING HOT! (popping like popcorn) IT'S ALL MINE! (can you hear the bells ding-dong?) Pick up every last one THIS IS IT! THIS IS IT! Let's not (pretend I don't hear me), Try to (move on without looking back). 'CAUSE I REALISED IIIIIIT! PERFECTION CAN'T PLEASE ME!! (WHOO-OO-HOO)!"
lyrList = lyrics.split(" ")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame,(480,852))

    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    #gray_frame = np.minimum((cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype('float64')*2),255).astype('uint8') 
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = detector.detectMarkers(gray_frame)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef#could be something wrong with MultiMatrix
        )
        total_markers = range(0, marker_IDs.size)
        # if marker_IDs.size > 1 or (marker_IDs.size == 1 and marker_IDs[0][0]!=999):
        #     print(marker_IDs)

        # List to store each marker's (x, z) position and ID
        positions = {}

        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            x, y, z = tVec[i][0]  # Extract x, y, z coordinates
            positions[ids[0]] = np.array([x,y,z])

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
            if(ids[0] == 999):
                cyl.setFramePos(rVec[i][0],tVec[i][0])
        
        cyl_colour = cyl.colour
        # Check if any two markers are aligned in (x, z) position

    if(cyl.overlaps(998,positions)):
        if(line_count < len(lyrList)):
            print(lyrList[line_count])
            line_count += 1
        cyl_colour = (0,255,100)

    cyl.recolour_draw(frame,cyl_colour)
    cv.imshow("frame", frame)
    out.write(frame)
    #recolour(over,(100,0,100))

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord('p'):
        cv.waitKey(-1) # Wait until any key is pressed
cap.release()
cv.destroyAllWindows()