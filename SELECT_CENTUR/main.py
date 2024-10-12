import cv2 as cv
from cv2 import aruco
import numpy as np

# Define the real width of the ArUco marker (in cm)
MARKER_REAL_WIDTH = 5.0  # Adjust this based on your actual marker size

# Load the ArUco marker dictionary
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Define ArUco detection parameters
param_markers = aruco.DetectorParameters()

# Focal length (in pixels). Adjust this value based on your camera's calibration.
FOCAL_LENGTH = 700  # You may need to calibrate your camera for the exact value.

# Open the camera feed from Iriun (or another IP webcam)
cup = cv.VideoCapture("http://192.168.1.20:8080/video")

# Verify camera opened correctly
if not cup.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cup.read()
    
    if not ret:
        print("Error: Failed to grab frame")
        break
    
    # Convert the frame to grayscale for ArUco detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect the ArUco markers in the grayscale frame
    marker_corners, marker_IDs, rejected = aruco.detectMarkers(
        gray_frame, 
        marker_dict, 
        parameters=param_markers
    )
    
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            # Draw polygon around the markers
            corners = corners.astype(np.int32)
            cv.polylines(frame, [corners], True, (0, 255, 255), 4, cv.LINE_AA)
            
            # Reshape and convert corner points to integers
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            
            # Top-right corner for displaying the marker ID
            top_right = corners[0].ravel()
            cv.putText(
                frame, 
                f"ID: {int(ids[0])}",  # Convert ID to integer for better display
                top_right, 
                cv.FONT_HERSHEY_PLAIN, 
                1.3, 
                (255, 255, 255), 2, cv.LINE_AA
            )
            
            # Calculate the center of the marker and draw a red circle
            center = np.mean(corners, axis=0).astype(int)
            cv.circle(frame, tuple(center), 5, (0, 0, 255), -1)
            
            # Calculate the distance between all four corners (side lengths)
            side_lengths = [
                np.linalg.norm(corners[0] - corners[1]),  # Top edge
                np.linalg.norm(corners[1] - corners[2]),  # Right edge
                np.linalg.norm(corners[2] - corners[3]),  # Bottom edge
                np.linalg.norm(corners[3] - corners[0])   # Left edge
            ]
            
            # Calculate the average side length in pixels
            avg_pixel_width = np.mean(side_lengths)
            
            # Calculate the distance to the marker
            distance = (MARKER_REAL_WIDTH * FOCAL_LENGTH) / avg_pixel_width
            
            # Display the distance on the frame
            cv.putText(
                frame, 
                f"Distance: {distance:.2f} cm",  # Display the distance in cm
                (top_right[0], top_right[1] + 20),  # Slightly below the marker ID
                cv.FONT_HERSHEY_PLAIN, 
                1.3, 
                (0, 255, 0), 2, cv.LINE_AA
            )
    
    # Display the frame with the detected markers and distance
    cv.imshow("ArUco Marker Detection with Distance", frame)
    
    # Break the loop on 'q' key press
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cup.release()
cv.destroyAllWindows()
