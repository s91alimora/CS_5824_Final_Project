import cv2
import pandas as pd
from pose_tracking import *

if __name__ == "__main__":
    label = 'pick'
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('C:\\Users\\srije\\Unity Projects\\Human form analysis\\Recordings\\movie_001.mp4')
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    dataset = []
    while(cap.isOpened()):
        success, frame = cap.read()

        if success:
            landmarks = detect_pose_landmarks(frame)

            if landmarks:
                dataset.append(landmarks_to_features(landmarks))
                draw_landmarks(frame,landmarks)

            frame = cv2.putText(frame, f'label: {label}', (25,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.6, color = (0, 255, 0), thickness = 2)
            # frame = cv2.putText(frame, f'#{i}', (frame.shape[1]-100, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                    fontScale=0.6, color=(0, 255, 0), thickness=2)
            cv2.imshow("Frame",frame)
            cv2.waitKey(1)
        else:
            break

    dataset = pd.DataFrame(dataset)
    # print(dataset)
    dataset.to_csv(f"{label}.csv",index=False)