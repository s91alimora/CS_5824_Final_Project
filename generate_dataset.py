import cv2
import pandas as pd
from pose_tracking import *
import os

if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)

    directory = "C:/Users/s_ali/OneDrive - Virginia Tech/Classes/" \
                "3- Spring 2022/CS 5824 - Adv ML/"
    ## 'C:/Users/srije/Unity Projects/Human form analysis/Recordings'

    ### Looping over two folders in the directory to
    ####       read the existing files and assign "right" or "wrong" label.
    folderList = os.listdir(directory)
    for folder in folderList:
        dataset = []
        folderPath = directory + folder
        dataLabel = folder
        for fileList in os.scandir(folderPath):
            if fileList.is_file():
                filePath = fileList.path
                cap = cv2.VideoCapture(filePath)
                if not cap.isOpened():
                    print("Error opening video stream or file")

                while cap.isOpened():
                    success, frame = cap.read()

                    if success:
                        landmarks = detect_pose_landmarks(frame)

                        if landmarks:
                            row = landmarks_to_features(landmarks)
                            row.append(dataLabel)
                            dataset.append(row)
                            draw_landmarks(frame, landmarks)

                        frame = cv2.putText(frame, f'label: {dataLabel}', (25, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.6, color=(0, 255, 0), thickness=2)
                        # frame = cv2.putText(frame, f'#{i}', (frame.shape[1]-100, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #                    fontScale=0.6, color=(0, 255, 0), thickness=2)
                        cv2.imshow("Frame", frame)
                        cv2.waitKey(1)
                    else:
                        break

    Dataset = pd.DataFrame(dataset)
    # print(dataset)
    Dataset.to_csv("Training_Data.csv", index=False)