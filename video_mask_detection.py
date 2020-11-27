from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import cv2

def main():
    tf.config.experimental.set_device_policy(None)

    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = load_model("./model/detection", compile=True)
    lb = pickle.loads(open("./model/labeler", "rb").read())

    stream = cv2.VideoCapture("./video/maskTest.mp4")
    writer = None
    (W, H) = None, None

    while True:
        #get next frame
        (present, frame) = stream.read()

        #if frame isnt avaliable then the video is over
        if not present:
            break
        
        #get frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))#.astype("float32")
        print(str(np.expand_dims(frame, axis=0)[0]))
        prediction = model.predict(np.expand_dims(frame, axis=0))[0]
        print(prediction)

if __name__ == "__main__":
    main()
