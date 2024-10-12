import cv2 as cv 
from cv2 import aruco

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

MARKER_SIZE =200 # pixels

for id in range(20):
    marker_img = aruco.drawMarker(marker_dict , id , MARKER_SIZE)
    # cv.imshow("test" , marker_img)
    cv.imwrite(f"markers/markder_{id}.png" , marker_img)
    # cv.waitKey(0)
    # break