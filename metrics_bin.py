import pickle

import numpy as np
import json
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import np_utils

model_name = sys.argv[1]
model = load_model("models/" + model_name + ".h5")


def loadFromPickle():
    with open("testfl/" + model_name + "_features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("testfl/" + model_name + "_labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


def get_eval(y_true, y_prob):
    output = {}
    print(y_prob.shape, y_true.shape)
    output['accuracy'] = metrics.accuracy_score(y_true, y_prob)
    output['classification_report'] = metrics.classification_report(y_true, y_prob)
    output['confusion_matrix'] = np.reshape(metrics.confusion_matrix(y_true, y_prob), (2, 2)).tolist()
    return output


features, labels = loadFromPickle()
features, labels = shuffle(features, labels)
labels=prepress_labels(labels)
train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=1,
                                                    test_size=0.2)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
valid_x = valid_x.reshape(test_x.shape[0], 28, 28, 1)

y_prob = model.predict_classes(test_x)
print(y_prob, np.argmax(test_y, axis=-1))
eval = get_eval(np.argmax(test_y, axis=-1), y_prob)
print("Accuracy is", eval["accuracy"])


print(eval['classification_report'])