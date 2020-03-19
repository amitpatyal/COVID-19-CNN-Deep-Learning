import os
import cv2
import pickle
import numpy as np
from keras.models import load_model

path = 'test_data' # test data set folder path
model = load_model('models/COVID-19-model.h5')
labels = pickle.load(open('models/labels.pkl','rb'))
labelsValues = ['Covid - 19', 'Normal']
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def getPrediction():
    predictionValue = ''
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.array(image) / 255.0
        image = image.reshape(224, 224, -1)
        image1 = np.expand_dims(image, axis=0)
        prediction = model.predict(image1)
        prediction = np.argmax(prediction, axis=1)
        predictionValue = np.int(labels[prediction[0].astype('int64')][0])
        if predictionValue == 0:
            cv2.putText(image, labelsValues[predictionValue], (50,40), font, 1,(255,0,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(image, labelsValues[predictionValue], (50, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('COVID-19: window', image)
        cv2.waitKey(500)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



if __name__ == '__main__':
    getPrediction()