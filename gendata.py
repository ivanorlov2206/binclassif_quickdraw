import os
import pickle
import random
import sys

# Put everything to data directory
import numpy as np
import subprocess

classes = ['airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm',
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


def download_data(name):
    subprocess.call(['gsutil', '-m', 'cp', "gs://quickdraw_dataset/full/numpy_bitmap/" + name + ".npy", './data/'])


def load_data(cln, st, e):
    count = 0
    bts = 0
    x_load = []
    y_load = []
    nl = ["squiggle.npy"]
    if cln != "square":
        nl.append("square.npy")
    if cln != "circle":
        nl.append("circle.npy")
    if cln != "line":
        nl.append("line.npy")
    random.shuffle(nl)
    nf = nl[:10] + [cln + ".npy"]
    for file in nf:
        cln2 = file.split('.')[0]
        file = "data/" + file
        x = np.load(file)
        x = x[st:e, :]
        x = x.astype('float32') / 255.
        x = (x > 0) * 1
        bts += sys.getsizeof(x)
        x_load.append(x)
        y = [1 if cln == cln2 else 0 for _ in range(e - st)]
        #print(file, y)
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


def create_fl(cln, dir, s, e):
    features, labels = load_data(cln, s, e)
    features = np.array(features).astype('float32')
    labels = np.array(labels).astype('float32')
    features = features.reshape(features.shape[0] * features.shape[1], features.shape[2])
    labels = labels.reshape(labels.shape[0] * labels.shape[1], labels.shape[2])
    with open(dir + "/" + cln + "_features", "wb") as f:
        print(features)
        pickle.dump(features, f, protocol=4)
    with open(dir + "/" + cln + "_labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)

n = 0
download_data("squiggle")
download_data("square")
download_data("line")
download_data("circle")
for class_name in classes[:200]:
    if not class_name in ["squiggle"]:
        if not os.path.isfile('data/{}.npy'.format(class_name.replace(' ', '_'))):
            print('Downloading {}...'.format(class_name))
            download_data(class_name)
            ocn = class_name
            class_name = class_name.replace(' ', '_')
            os.rename('./data/{}.npy'.format(ocn), './data/{}.npy'.format(class_name))
        else:
            class_name = class_name.replace(' ', '_')
        print("Creating features and labels...")
        create_fl(class_name, 'fldata', 0, 20000)
        if not class_name in ["square", "circle"]:
            os.remove('./data/{}'.format(class_name + ".npy"))

'''for fn in files:
    cln = fn.split('.')[0]
    create_fl(cln, 0, 10000)
'''