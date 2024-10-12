import cv2 as cv
import numpy as np

imag = cv.imread("images/img_1.jpg")

cv.putText( img=imag,text="Test",
            org=(30,40),
            fontFace=cv.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0,255,0)
            )

cv.imshow("img" , imag)
cv.waitKey(0)
cv.destroyAllWindows()
