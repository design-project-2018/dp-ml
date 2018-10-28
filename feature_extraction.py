from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten

import cv2
import numpy as np

def build_VGG_model():
    # Initialize VGG16 model using pretrained ImageNet weights without last 3 fc layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    print('Initializing VGG16 model with ImageNet weights...')

    # Initialize two fc layers
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(4096, activation='relu'))
    top_model.add(Dense(4096, activation='relu'))
    print('Creating fully connected layers of size 4096...')

    # Copy VGG layers in order to concatenate
    new_model = Sequential()
    for layer in base_model.layers:
        new_model.add(layer)

    new_model.add(top_model)

    # Make the layers untrainable
    for layer in new_model.layers:
        layer.trainable = False
    print('VGG16 Feature Extraction Network Complete!')

build_VGG_model()