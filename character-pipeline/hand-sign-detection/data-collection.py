import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, frame = cap.read()
    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[1]
        x, y, w, h = hand['bbox']

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

