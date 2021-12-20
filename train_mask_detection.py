from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4 
EPOCHS = 20  
BS = 32 


DIR = r"/Users/jaketaranov/Desktop/mask_detection/dataset"
CATEGOREIES = ["with_mask", "without_mask"]

data = []
labels = []
#reading in data and formatting it
for category in CATEGOREIES:
    path = os.path.join(DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)
    
lb = LabelBinarizer()
labels = lb.fit_transform(labels) #scale images 
labels = to_categorical(labels) #puts data in a matrix


data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15, stratify=labels, random_state=35) #splitting data up into subsets 

augmented = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,shear_range=0.15,horizontal_flip=True, fill_mode="nearest") #adjusting images for diversity in dataset 



baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))) #creating a base model, imagenet is a alread trained base model for images, we will add a top layer later, 
                                                                                                                                                        #  input tensor is the size of the image we are passing in (3 channels)


#creating the neural net 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel) # relu activation function (best for images)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) #softmax is a general use activation function

model = Model(inputs=baseModel.input, outputs=headModel) # 

for layer in baseModel.layers:
    layer.trainable = False    #deactivate all layers in the base model so they will not be updated during the first iteration

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) #Adam optimizer, (best for images)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#training head
H = model.fit(
	augmented.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#evaluating 
predictedIndexes = model.predict(testX, batch_size = BS)
#
predictedIndexes = np.argmax(predictedIndexes, axis=1)

print(classification_report(testY.argmax(axis=1), predictedIndexes, target_names=lb.classes_))

model.save("mask_detector_EPOCH20.model", save_format="h5")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")





