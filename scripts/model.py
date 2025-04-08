import pandas as pd
import numpy as np
import re
import emoji


from string import punctuation
from os import listdir
from collections import Counter
from string import punctuation
from os import listdir
from numpy import array
import tensorflow
import keras
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, RNN, SimpleRNN, ConvLSTM1D
from keras.layers import Flatten, Dropout
from keras.layers import Embedding, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def model(train_x, train_y, valid_x, valid_y, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=20))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
    # model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=3, verbose=2)
    return history

def test(model, test_x, test_y):
    loss, acc = model.evaluate(test_x, test_y, verbose=0)
    print('Test Accuracy: %f' % (acc*100))

def predict_sample(text, tokenizer, encoded_docs, maxlen=20):
    encoded_doc = tokenizer.texts_to_sequences(text)
    test = pad_sequences(encoded_docs, maxlen, padding='post')
    prediction = model.predict(test)
    predictions = (model.predict(test) > 0.5).astype("int32")
    print('Prediction value, pre-rounding: {:.2f}'.format(prediction[0][0]))
    if predictions[0][0] == 0:
        print("Negative post detected")
    else:
        print("Negative post detected:")
