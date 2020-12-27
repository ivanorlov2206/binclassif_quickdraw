import os

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.layers import MaxPooling2D, Dropout
from tensorflow.python.keras.utils import np_utils
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep

files = ['airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm',
           'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket',
           'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle',
           'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie',
           'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush',
           'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire',
           'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair',
           'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer',
           'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond',
           'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums',
           'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan',
           'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight',
           'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden',
           'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand',
           'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick',
           'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant',
           'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern',
           'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
           'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
           'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom',
           'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush',
           'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas',
           'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza',
           'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit',
           'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'river', 'roller coaster',
           'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion',
           'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard',
           'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman',
           'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel',
           'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry',
           'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword',
           't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent',
           'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
           'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone',
           'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon',
           'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra',
           'zigzag']

def keras_model(image_x, image_y, name):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "models/{}.h5".format(name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle(name):
    with open("fldata/{}_features".format(name), "rb") as f:
        features = np.array(pickle.load(f))
    with open("fldata/{}_labels".format(name), "rb") as f:
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
    y_pred = np.argmax(y_prob, -1)
    output = {}
    output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    output['loss'] = metrics.log_loss(y_true, y_prob)
    output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def main():
    for f in files:
        if f == "squiggle":
            continue
        f = f.replace(' ', '_')
        if not os.path.isfile("models/{}.h5".format(f)):
            print("Training on", f)
            model, callbacks_list = keras_model(28, 28, f)
            features, labels = loadFromPickle(f)
            features, labels = shuffle(features, labels)
            #labels=prepress_labels(labels)
            print(labels.shape)
            train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=1,
                                                                test_size=0.2)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
            train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
            test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
            valid_x = valid_x.reshape(test_x.shape[0], 28, 28, 1)
            model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=5, batch_size=64,
                      callbacks=[TensorBoard(log_dir="QuickDraw")])

            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

            # Compute end step to finish pruning after 2 epochs.
            batch_size = 128
            epochs = 2
            validation_split = 0.1  # 10% of training set will be used for validation set.

            num_images = train_x.shape[0] * (1 - validation_split)
            end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

            # Define model for pruning.
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                         final_sparsity=0.90,
                                                                         begin_step=0,
                                                                         end_step=end_step)
            }

            model_for_pruning = prune_low_magnitude(model, **pruning_params)

            # `prune_low_magnitude` requires a recompile.
            model_for_pruning.compile(optimizer='adam',
                                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                      metrics=['accuracy'])

            model_for_pruning.summary()

            model_for_pruning.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=3, batch_size=64,
                      callbacks=[UpdatePruningStep()])

            fin_model = strip_pruning(model_for_pruning)


            tf.keras.models.save_model(fin_model, 'models/{}.h5'.format(f), include_optimizer=False)


main()
