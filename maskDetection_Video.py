from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2 as cv
import imutils
import time
import os


def detection(frame, faceNet, maskNet):
    faces = []
    locations = []
    preditions = []

    (h,w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1, (224,224), 104, 177, 123)
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0,startX), max(0,startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))
        
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preditions = maskNet.predict(faces, batch_size=32)
    return(locations, preditions)


weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
protoPath = r"face_detector/deploy.prototxt"
maskNet = load_model("mask_detector_EPOCH20.model")
faceNet = cv.dnn.readNet(protoPath, weightsPath)

vid = VideoStream(src=0).start()
time.sleep(2)

while True:
    count = 0
    frame = vid.read()
    frame = imutils.resize(frame, width=1000, height=1000)
    (locations, predictions) = detection(frame, faceNet, maskNet)
    for(box, prediction) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (mask, wo_mask) = prediction
        if mask > wo_mask:
            label = "Mask"
            color = (0,255,0)
        else:
            label = "No Mask"
            color = (0,0,255)
                
        label = "{}: {:.2f}%".format(label, max(mask, wo_mask) * 100)

        #cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_COMPLEX, 1, [color], 2)
        cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_COMPLEX, 1, [0,0,0], thickness=4)
        cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_COMPLEX, 1, color, lineType = cv.LINE_AA, thickness=2)
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv.imshow("Mask Detector", frame)
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
vid.stop()


