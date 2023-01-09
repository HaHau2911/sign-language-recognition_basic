import pickle
import cv2
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

model = tf.keras.models.load_model('./asl3.h5')
with open('labels', 'rb') as f:
    labels = pickle.load(f)
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    frame = cv2.rectangle(frame, (350, 50), (600, 300),
                          (255, 0, 0), thickness=2)
    img_pred = frame[50:300, 350:600, :]
    img_pred = cv2.resize(img_pred, (200, 200))

    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    img_pred = np.expand_dims(img_pred, axis=0)
    predict = model.predict(img_pred)[0]
    index = np.argmax(predict)
    frame = cv2.putText(
        frame, 'Result: '+labels.inverse_transform([index])[0] + ' %.2f' % predict[index], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, thickness=2, color=(0, 0, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
