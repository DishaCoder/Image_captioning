import image as image
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pickle
import re
import string
import os
import glob
from PIL import Image
from time import time
from imutils import paths

from keras import Input, layers
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout,AveragePooling2D
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.metrics import Accuracy
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from keras.preprocessing import image

from keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import load_model

def extract_features(directory):
    base_model = InceptionV3(weights='imagenet')
    # to remove the last layer
    base_model.layers.pop()
    model_new = Model(base_model.input, base_model.layers[-2].output)
    print(model_new.summary())
    features = dict()
    if (os.path.isdir(directory)):
        for name in tqdm(os.listdir(directory)):
            try:
                filename = directory + '/' + name
                # print(filename)
                # Convert all the images to size 299x299 as expected by the inception v3 resources
                image = load_img(filename, target_size=(299, 299))
                # image = load_img(filename, target_size=(224, 224))
                # Convert PIL image to numpy array of 3-dimensions
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                # preprocess the images using preprocess_input() from inception module
                image = preprocess_input(image)
                # Get the encoding vector for the image
                feature = model_new.predict(image, verbose=0)
                image_id = name.split('.')[0]
                features[image_id] = feature
                # print('>%s' % name)
            except Exception as e:
                print(e)
                continue
    else:
        filename = directory
        print(filename)
        try:
            image = load_img(filename, target_size=(299, 299))
            print(image)
            # image = load_img(filename, target_size=(224, 224))
        except Exception as e:
            print(e)
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        print(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model_new.predict(image, verbose=0)
        image_id = filename.split('.')[0]
        features[image_id] = feature
        # print('>%s' % filename)
    return features


def extract_features1(directory):
    base_model = InceptionV3(weights='imagenet')
    # to remove the last layer
    base_model.layers.pop()
    model_new = Model(base_model.input, base_model.layers[-2].output)
    print(model_new.summary())

    if (os.path.isdir(directory)):
        for name in tqdm(os.listdir(directory)):
            try:
                filename = directory + '/' + name
                # print(filename)
                # Convert all the images to size 299x299 as expected by the inception v3 model
                img = load_img(filename, target_size=(299, 299))
                # Convert PIL image to numpy array of 3-dimensions
                x = img_to_array(img)
                # Add one more dimension
                x = np.expand_dims(x, axis=0)
                # preprocess the images using preprocess_input() from inception module
                x = preprocess_input(x)
                fea_vec = model_new.predict(x)  # Get the encoding vector for the image
                fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )

                # print('>%s' % name)
            except Exception as e:
                print(e)
                continue
    else:
        filename = directory
        print(filename)
        try:
            img = load_img(filename, target_size=(299, 299))
            print(image)
            # image = load_img(filename, target_size=(224, 224))
        except Exception as e:
            print(e)

        # Convert all the images to size 299x299 as expected by the inception v3 model
        img = load_img(filename, target_size=(299, 299))
        # Convert PIL image to numpy array of 3-dimensions
        x = image.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess the images using preprocess_input() from inception module
        x = preprocess_input(x)
        fea_vec = model_new.predict(x)  # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )

    return fea_vec


def creat_index_to_word_viceversa(vocab):
  indexto_word = {}
  wordto_index = {}
  index = 1
  for word in vocab:
      wordto_index[word] = index
      indexto_word[index] = word
      index += 1
  return wordto_index,indexto_word


train_vocab = pickle.load(open("D:/lambton/Projects/image_caption_generator1/resources/train_vocab.pkl","rb"))
train_wordto_index,train_indexto_word = creat_index_to_word_viceversa(train_vocab)
max_length = 34

train_vocab = pickle.load(open("D:/lambton/Projects/image_caption_generator1/resources/train_vocab.pkl", "rb"))

# Model Loading
# model = load_model("D:/lambton/Projects/image_caption_generator1//resources/my_model.h5")
model = load_model("D:/lambton/Projects/image_caption_generator1//resources/my_model_new.h5")


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [train_wordto_index[w] for w in in_text.split() if w in train_wordto_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = train_indexto_word[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final



