import cv2 as cv 
from cv2 import aruco
import numpy as np

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

cup = cv.VideoCapture(0)

while True:
    ret , frame = cup.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    marker_corners , marker_IDs , reject = aruco.detectMarkers(
        gray_frame,
        marker_dict,
        parameters=param_markers
    )
    if marker_corners :
        for ids ,corners in zip(marker_IDs , marker_corners):
            cv.polylines(frame , [corners.astype(np.int32)] ,True ,(0,255,255) , 4 , cv.LINE_AA)
            # print(ids , " " , corners) 
            corners = corners.reshape(4,2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            cv.putText( frame,
                       f"ID : {ids[0]}",
                        top_right,
                        cv.FONT_HERSHEY_PLAIN,
                        1.3,
                        (200,100,0),
                        2,
                        cv.LINE_AA
                    )

    cv.imshow("frame",frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cup.release() 
cv.destroyAllWindows()