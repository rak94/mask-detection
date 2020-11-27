"""
Name: Mask Detection
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import layers, models
import numpy as np
import os
import pickle
import tensorflow as tf

def main():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = (tf.compat.v1.Session(config=config))

    #Gather data from dataset
    data, labels = gather_images('dataset')

    #One hot encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    #labels = to_categorical(labels)

    #Split into training and test sets 80/20
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels)

    #Build model
    model = models.Sequential()
    model.add(layers.Conv2D(25,(5, 5), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(70, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(70, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    #Compile and train model
    opt = optimizers.SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=trainX, y=trainY, epochs=5)

    #Evaluate and print accuracy
    test_loss, test_acc = model.evaluate(testX, testY)
    print(test_acc)

    print("Saving Model")
    model.save("./model/detection")
    f = open("./model/labeler", "wb")
    f.write(pickle.dumps(lb))
    f.close()

def file_paths(dataset_dir):
    """
    Gathers a list of jpg image paths in all subdirectories of given directory. 
    """
    for (root, dirNames, fileNames) in os.walk(os.getcwd() + '/' + dataset_dir):
        for fileName in fileNames:
            ext = fileName[fileName.rfind("."):].lower()
            if (ext.endswith(".jpg")):
                imagePath = os.path.join(root, fileName)
                yield imagePath

def gather_images(dataset_dir):
    """
    Gathers images and labels in two seperate numpy arrays
    """
    image_paths = list(file_paths(dataset_dir))
    data = []
    labels = []
    
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)

    #Convert to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    main()