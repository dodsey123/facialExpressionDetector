# need to run pip install opencv-contrib-python
import cv2
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
from PIL import Image

class GetImage:
    def __init__(self):
        self.runGetImage()

    def runGetImage(self):
        cascPath = os.path.abspath(os.path.dirname(os.getcwd())) + "/haarcascade_frontalface_alt.xml"
        print(os.path.exists(cascPath))
        faceCascade = cv2.CascadeClassifier(cascPath)

        model = load_model("emotion_model.keras")

        video_capture = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            _, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            faces = faceCascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )



            # Draw a rectangle around the faces
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)        
                
                # Extract the face ROI (Region of Interest)
                #face_roi = rgb[y:y + h, x:x + w]

                face = gray[y:y+h, x:x+w]

                # Ensure face is valid and resize it
                if face.size > 0:
                    # Resize the face image to the target size
                    face_resized = cv2.resize(face, (48, 48))  # Resize to (48, 48)

                    # Add an additional dimension for channels (1 channel for grayscale)
                    face_input = face_resized.reshape(1, (48, 48)[0], (48, 48)[1], 1)  # Shape (1, 48, 48, 1)

                    # Normalize the image
                    face_input = face_input / 255.0

        
                    # Perform emotion analysis on the face ROI
                    result = self.findEmotion(model.predict(face_input))
                    #print(result)

                # Determine the dominant emotion
                #emotion = result[0]['dominant_emotion']

                    # Draw rectangle around face and label with predicted emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display the resulting frame
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


    def findEmotion(self, emotionsInt):
        index = np.argmax(emotionsInt)
        emotions = ["suprise", "sadness", "happy", "fear", "disgust", "contempt", "anger"]
        return emotions[index]
