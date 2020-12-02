"""
Name: Mask Detection
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import layers, models, applications
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import pickle
import tensorflow as tf
import time

def main():
    start_time = time.time()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    session = (tf.compat.v1.Session(config=config))

    print("GATHERING DATA")
    #Gather data from dataset
    data, labels = gather_images('self-built-masked-face-recognition-dataset')
    print("DONE:--- %s seconds ---\n" % (time.time() - start_time))
    
    print("LABELING DATA")
    #One hot encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    print("DONE:--- %s seconds ---\n" % (time.time() - start_time))

    #Split into training and test sets 80/20
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels)
    
    #Simple Sequential Model
    """
    #Build model
    model = models.Sequential()

    model.add(layers.Conv2D(25,(3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2,2)))

    model.add(layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    for i in range(1,5):
        print("COMPILING AND TRAINING WITH {0} EPOCHS".format(i))
        #Compile and train model
        opt = optimizers.Adam(lr=1e-3) #decay=(1e-3)/10)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=trainX, y=trainY, epochs=i)
        print("DONE:--- %s seconds ---\n" % (time.time() - start_time))

        #Evaluate and print accuracy
        test_loss, test_acc = model.evaluate(testX, testY)
        print(test_acc)

        print("Saving Model {0} Epochs".format(i))
        file_name = "./model/detection_epochs_{0}.h5".format(i)
        model.save(file_name)
    """
    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    baseModel = applications.VGG16(weights="imagenet", include_top=False,
        input_tensor=layers.Input(shape=(128, 128, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(2, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = models.Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    for i in range(1,6):
        # compile our model
        print("Training model epoch {0}".format(i))
        opt = optimizers.Adam(lr=1e-4)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        # train the head of the network
        model.fit(x=trainX, y=trainY,epochs=1)

        # make predictions on the testing set
        predIdxs = model.predict(testX)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)

        # show a nicely formatted classification report
        print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

        # serialize the model to disk
        print("Saving Model {0} Epochs".format(i))
        file_name = "./model/detection_vgg16_epochs_{0}.h5".format(i)
        model.save(file_name)

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
        label = image_path.split(os.path.sep)[-3]
        
        image = load_img(image_path, target_size=(128, 128))
        image = img_to_array(image)
        #image = preprocess_input(image)

        data.append(image)
        labels.append(label)

    #Convert to numpy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels

if __name__ == "__main__":
    main()