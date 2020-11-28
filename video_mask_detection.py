import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import sys
import pickle
import cv2
from video_frame_helper import videoFrameHelper

def main():
    tf.config.experimental.set_device_policy(None)
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = load_model("./model/detection.h5", compile=True)
    lb = pickle.loads(open("./model/labeler", "rb").read())

    #Face Detection Loading
    prototxtPath = "./model/face_detector/deploy.prototxt"
    weightsPath = "./model/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceDetectionModel = cv2.dnn.readNet(prototxtPath, weightsPath)

    stream = cv2.VideoCapture("./video/maskTest.mp4")
    helper = videoFrameHelper(stream)
    
    writer = None
    (W, H) = None, None

    while True:
        #get next frame
        frame = helper.getNextFrame()
        
        #get frame dimensions
        if (W is None or H is None):
            (H, W) = frame.shape[:2]
        
        pred = predict_label(faceDetectionModel, model, W, H, frame)
        if (len(pred) < 0):
            (mask, no_mask) = pred
            label = "Mask" if mask > no_mask else "No Mask"
        else:
            label = "No Mask"
        
        #output = frame.copy()
        text = "activity:{0}".format(label)
        cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
        """
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("./output_video/mask_detect_video.avi", fourcc, 30, (W, H), True)

        writer.write(output)
        """
        cv2.imshow("Frame", frame)
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    print("Clean up worker thread")
    cv2.destroyAllWindows()
    helper.terminateThread()

def predict_label(faceModel, maskModel, W, H, frame):
        #uncomment for face detection then mask detection over the that was detected
        """
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104,0, 177.0, 123.0))
        faceModel.setInput(blob)
        potentialFaces = faceModel.forward()

        prediction = 0
        faces = []
        predictions = []

        for i in range(0, potentialFaces.shape[2]):
            probability = potentialFaces[0,0,i,2]

            if (probability*100 > 95):
                faceArea = potentialFaces[0,0,i,3:7] * np.array([W,H,W,H])
                (xOne, yOne, xTwo, yTwo) = faceArea.astype("int")

                face = frame[yOne:yTwo, xOne:xTwo]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0)
                faces.append(face)

        #if(len(faces) > 0):
            predictions = maskModel.predict(faces, batch_size=32)
        """

        #uncomment theses lines for mask detection only
        face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        predictions = maskModel.predict(face, batch_size=32)
        
        #Do not comment theses out
        print(predictions)
        return predictions

if __name__ == "__main__":
    main()
