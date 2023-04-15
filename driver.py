from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from gtts import gTTS
import os
import platform
import argparse
import imutils
import time
import dlib
import math
from cv2 import cv2
import numpy as np
from ear import eye_aspect_ratio
from mar import mouth_aspect_ratio

# Init dlib Face Detetctor | Create Facial Landmark predictor

print("Loading the Facial Landmark predictor from dlib")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/facial_landmarks.dat')

# Init camera
alert = "Hey it looks like you're tired ! Why don't you take a quick break ?"
voice = gTTS(text = alert, lang = "en", slow = False)
voice.save("fatigue.mp3")
print("Initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
frame_width = 1024
frame_height = 576

# To loop over frames from the stream
image_points = np.array([
    (359, 391),     # Nose tip (IDX=34)
    (399, 561),     # Chin (IDX=9)
    (337, 297),     # Left Eye Left Corner (IDX=37)
    (513, 301),     # Right Eye Right Corner (IDX=46)
    (345, 465),     # Left Mouth Corner  (IDX=49)
    (453, 469)      # Right Mouth Corner (IDX=55)
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_THRESHOLD = 0.27
MOUTH_THRESHOLD = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# Indices for Facial Landmarks for mouth
(mStart, mEnd) = (49, 68)

while True:
    # Grab & Resize the frame, then convert to GrayScale
    frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Checking if a face was detected and drawing that over the frame
    if len(rects) > 0:
        text = f"{len(rects)} Face(s) found...."
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Looping over each face in the case of multiple faces..
    for rect in rects:
        # Calculating the bounding box for each face and drawing that over the frame.
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # Determine facial landmarks and convert the points to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract coordiantes for left eye and right eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # Calculating aspect ratios for each eye
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Drawing the convex hull for each eye.
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Comparing ear with threshold
        if ear < EYE_THRESHOLD:
            COUNTER += 1
            # Display warning based on no:of times 
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes are Closed", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if platform.system() == "Windows":
                #     os.system("mpg123 " + "fatigue.mp3")
                    
                # else:
                #     os.system("afplay " + "fatigue.mp3" )
                    
        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]
        # Getting the mouth aspect ratio
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw text if mouth is open
        if mar > MOUTH_THRESHOLD:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if platform.system() == "Windows":
                    os.system("mpg123 " + "fatigue.mp3")
                    #time.sleep(10)
            else:
                    os.system("afplay " + "fatigue.mp3" )
                    #time.sleep(10)
            


        
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()