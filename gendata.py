import os
import pickle
import sys

# Put everything to data directory
import numpy as np

files = os.listdir("data/")


def load_data(cln):
    count = 0
    bts = 0
    s = set(files)
    s.remove(cln + ".npy")
    nf = list(s)[:3] + [cln + ".npy"]
    for file in nf:
        cln2 = file.split('.')[0]
        file = "data/" + file
        x = np.load(file)
        x = x[0:10000, :]
        x = x.astype('float32') / 255.
        x = (x > 0) * 1
        bts += sys.getsizeof(x)
        x_load.append(x)
        y = [1 if cln == cln2 else 0 for _ in range(10000)]
        if (cln == cln2):
            print(1212121212)
        #print(file, y)
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


for fn in files:
    cln = fn.split('.')[0]

    x_load = []
    y_load = []
    print([file[:file.find('.')] for file in files])
    features, labels = load_data(cln)
    features = np.array(features).astype('float32')
    labels = np.array(labels).astype('float32')
    features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
    labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])
    with open("fldata/" + cln + "_features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("fldata/" + cln + "_labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)