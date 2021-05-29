import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX('densenet121.onnx')
classNames = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frameWidth = 480
frameHeight = 500
lightLevel = 130

cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,lightLevel)

while True:
    success, img = cap.read()
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (224, 224), (104, 117, 123), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    biggest_pred_index = np.array(preds)[0].argmax()
    cv2.putText(img, classNames[biggest_pred_index].upper(), (10 + 10, 10 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

    cv2.imshow("Output", img)
    if not success:
        print("Failed to grab frame")
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
