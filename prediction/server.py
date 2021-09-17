import socket
import os
from _thread import *

import json
from os import listdir
from os.path import isfile, join, exists, isdir, abspath
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

import datetime

IMAGE_DIM = 224

script_directory=os.path.dirname(__file__)
model_directory=join(os.path.dirname(script_directory), 'models')

def load_images(image_paths, image_size, verbose=True):
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]


    for img_path in image_paths:
        if not exists(img_path):
            print('File not exist: ', img_path)
            break
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)

    return np.asarray(loaded_images), loaded_image_paths

def load_model(model_path):
    if model_path is None or not exists(model_path):
        raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    if images.size > 0:
        probs = classify_nd(model, images)
    else:
        probs = []

    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images):
    model_preds = model.predict(nd_images)
    class_names=[]
    with open(model_directory + '/labels.txt') as file:
        class_names=file.readlines()
        class_names=[line.rstrip() for line in class_names]

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[class_names[j]] = float(pred)
        probs.append(single_probs)
    return probs

def threaded_client(connection):
    connection.send(str.encode('\n\nWelcome! Enter image path to scan\n# Path: '))
    while True:
        data = connection.recv(2048)
        reply = '> Scanning: ' + data.decode('utf-8')
        if not data:
            break

        filename = data.decode('utf-8').rstrip();
        connection.sendall(str.encode(reply))

        start = datetime.datetime.now()
        image_preds = classify(model, filename, IMAGE_DIM)
        end = datetime.datetime.now()

        delta = end - start
        image_preds['__time__'] = delta.seconds + (delta.microseconds / 1000000)
        reply = '> Result for : ' + data.decode('utf-8') + '\n' + json.dumps(image_preds, indent=2) + '\n\n# Path: '

        connection.sendall(str.encode(reply))
    connection.close()


print('Loading model...\n')
model = load_model(model_directory)    
print('\nModel loaded. Starting server...')

ServerSocket = socket.socket()
ServerSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = '0.0.0.0'
port = 1235
ThreadCount = 0
try:
    ServerSocket.bind((host, port))
except socket.error as e:
    print('Socket error: {}'.format(str(e)))

print('Server Ready. Listening at {}:{}'.format(host, port))
ServerSocket.listen(5)

while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))

ServerSocket.close()