from landmarks import HandLandmark
import cv2
import numpy as np

handLandmark = HandLandmark(max_num_hands=2, min_detection_confidence=0.5, min_landmark_confidence=0.5)
webcam = cv2.VideoCapture(0)
prev_points = {}
canvas = None

while True:
    status, frame = webcam.read()
    if not status:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    handLandmarks = handLandmark.findHandLandmarks(image=frame, draw=True)

    for hand_id, hand in enumerate(handLandmarks):
        index_4 = None
        index_8 = None
        for landmark_point in hand:
            if landmark_point[0] == 4:
                index_4 = (landmark_point[1], landmark_point[2])
            elif landmark_point[0] == 8:
                index_8 = (landmark_point[1], landmark_point[2])
                
        if index_4 and index_8:
            midpoint = ((index_4[0] + index_8[0]) // 2, (index_4[1] + index_8[1]) // 2)
            if hand_id in prev_points:
                prev_index_4, prev_index_8 = prev_points[hand_id]
                prev_midpoint = ((prev_index_4[0] + prev_index_8[0]) // 2, (prev_index_4[1] + prev_index_8[1]) // 2)
                distance_4_8 = ((index_4[0] - index_8[0]) ** 2 + (index_4[1] - index_8[1]) ** 2) ** 0.5
                if distance_4_8 < 30:
                    cv2.line(canvas, prev_midpoint, midpoint, (0, 0, 255), 5)
            prev_points[hand_id] = (index_4, index_8)

    frame = cv2.add(frame, canvas)
    cv2.imshow("Hand Landmark", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
webcam.release()
