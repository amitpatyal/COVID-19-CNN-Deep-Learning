import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

path = 'dataset' # data set folder path
#initialize the learning rate , epochs and batch size for trainig.
learningRate = 1e-3
epochs = 25
batchSize = 8
imageData = []
labels = []

imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
count = 0
print("[INFO] loading images...")
for imagePath in imagePaths:
    images = [os.path.join(imagePath, f) for f in os.listdir(imagePath)]
    for image in images:
        labels.append(imagePath.split(os.path.sep)[-1])  # [-1] get the subfolder name and [-2] get the main folder name.
        count += 1
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        imageData.append(image)

print("Total:{:.0f} images are processed ".format(count) +"and:{:.0f} labels are found.".format(np.size(np.unique(labels))))

#one-hot encoding for labels
imageData = np.array(imageData) / 255.0
labels = np.array(labels)
labelBinarizer = LabelBinarizer()
labels = labelBinarizer.fit_transform(labels)
labels = to_categorical(labels)
pickle.dump(labels, open('models/labels.pkl', 'wb'))

(xTrain, xTest, yTrain, yTest) = train_test_split(imageData, labels, test_size=0.20, stratify=labels, random_state=42)
# initialize the training data augmentation object
trainAugmentation = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

#VGG16 Network
baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

optimizer = Adam(lr=learningRate, decay=np.floor_divide(learningRate, epochs))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#modelResult = model.fit(xTrain, yTrain,  steps_per_epoch=len(xTrain) // batchSize,
#          validation_data=(xTest, yTest), validation_steps=len(xTest // batchSize), epochs=epochs)
modelResult = model.fit_generator(trainAugmentation.flow(xTrain, yTrain, batch_size=batchSize), steps_per_epoch=np.floor_divide(len(xTrain), batchSize),
                                  validation_data=(xTest, yTest), validation_steps=np.floor_divide(len(xTest), batchSize), epochs=epochs)

print("[INFO] evaluating network...")
modelPredict = model.predict(xTest, batch_size=batchSize)
modelPredict = np.argmax(modelPredict, axis=1)

# show a nicely formatted classification report
print(classification_report(yTest.argmax(axis=1), modelPredict,	target_names=labelBinarizer.classes_), 'Classification Report')

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
confusionMatrix = confusion_matrix(yTest.argmax(axis=1), modelPredict)
total = sum(sum(confusionMatrix))
accuracy = (confusionMatrix[0, 0] + confusionMatrix[1, 1]) / total
sensitivity = confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
specificity = confusionMatrix[1, 1] / (confusionMatrix[1, 0] + confusionMatrix[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(confusionMatrix, 'Confusion Matrix')
print("Model accuracy: {:.4f}".format(accuracy) + " Model sensitivity: {:.4f}".format(sensitivity)+ " Model specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), modelResult.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), modelResult.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), modelResult.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), modelResult.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('model.png')

model.save('models/COVID-19-model.h5', overwrite=True)
print("[INFO] saving COVID-19 detector model...")
