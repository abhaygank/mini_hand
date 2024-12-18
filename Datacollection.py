import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Video capture setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 10  # Padding around the hand
imgSize = 200  # Image size for resizing

# Folder path where images will be saved
folder = r"C:\Users\ajayg\OneDrive\Desktop\worddata\WHICH"  # Use raw string for folder path
counter = 0

while True:
    success, img = cap.read()  # Read the current frame
    hands, img = detector.findHands(img)  # Detect hands in the frame

    if hands:  # If hands are detected
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box for the hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White canvas for resizing
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand region from the frame

        if imgCrop.size != 0:  # Ensure the crop is not empty
            aspectRatio = h / w
            if aspectRatio > 1:  # If height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize the image to match the aspect ratio
                wGap = (imgSize - wCal) // 2  # Calculate the horizontal gap for padding
                imgWhite[:, wGap:wCal + wGap] = imgResize  # Place the resized image onto the white canvas
            else:  # If width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the image to match the aspect ratio
                hGap = (imgSize - hCal) // 2  # Calculate the vertical gap for padding
                imgWhite[hGap:hCal + hGap, :] = imgResize  # Place the resized image onto the white canvas

            cv2.imshow("imageCrop", imgCrop)  # Show cropped image
            cv2.imshow("imageWhite", imgWhite)  # Show resized image

    cv2.imshow("image", img)  # Show the original image from the webcam

    key = cv2.waitKey(1)  # Wait for keypress
    if key == ord("s"):  # If 's' key is pressed, save the image
        counter += 1
        filename = f"{folder}/Image_{time.time()}.jpg"  # Save with timestamp
        cv2.imwrite(filename, imgWhite)  # Save the image
        print(f"Saved {counter} images")

# Release the webcam and close windows (you can use cv2.destroyAllWindows() when done)
cap.release()
cv2.destroyAllWindows()
