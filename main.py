import cv2
import stack
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
frameWidth = 480
frameHeight = 320
lightLevel = 130

cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, lightLevel)


def empty():
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2","Parameters",255,255,empty)


def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if area > 2000:
        cv2.drawContours(imgContour, contours, -1, (0, 255, 255), 7)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3)

while True:
    success, img = cap.read()
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
    # threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")

    threshold1 = 200
    threshold2 = 248

    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=1)

    getContours(imgDil,imgContour)

    imgStack = stack.stackImages(0.8, ([img, imgBlur, imgGray],[imgCanny,imgDil,imgContour]))

    cv2.imshow("Output", imgStack)
    if not success:
        print("Failed to grab frame")
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
