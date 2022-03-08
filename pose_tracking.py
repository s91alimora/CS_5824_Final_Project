import mediapipe as mp
import cv2

def landmarks_to_features(landmarks):
    features = []
    for id, lm in enumerate(landmarks.landmark):
        if id == 0:
            nose_x, nose_y, nose_z = lm.x, lm.y, lm.z

        #subtract wrist coordinates to make the model translation-invariant
        features.append(lm.x - nose_x)
        features.append(lm.y - nose_y)
        features.append(lm.z - nose_z)

    return features

poseDetector = mp.solutions.pose.Pose()
def detect_pose_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 produces images in BGR format, mediapipe uses only RGB
    pose_landmarks = None
    results = poseDetector.process(frame_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks  # read first hand

    return pose_landmarks

def draw_landmarks(frame,landmarks):
    return mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
