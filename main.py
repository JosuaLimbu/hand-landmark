from landmarks import HandLandmark
import cv2

handLandmark = HandLandmark(max_num_hands=2, min_detection_confidence=0.5, min_landmark_confidence=0.5)

webcam = cv2.VideoCapture(0)

while True:
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    handLandmarks = handLandmark.findHandLandmarks(image=frame, draw=True)

    for hand in handLandmarks:
        for landmark_point in hand:
            cv2.circle(frame, (landmark_point[1], landmark_point[2]), 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Hand Landmark", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
webcam.release()
