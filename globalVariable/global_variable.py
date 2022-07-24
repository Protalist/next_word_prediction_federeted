import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin")

import tensorflow as tf



import flwr as fl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense,Bidirectional,Layer, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam




    #from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Input,Embedding,
import pickle
import numpy as np

import string
import sklearn
import sklearn.model_selection
import math
import matplotlib.pyplot as plt
import nltk
import random
from functools import reduce
from attention import Attention

token_path=r"model\token\tokenizer2.pkl"
model_path=r"model\nextword_federeted.h5"
lengt_sequence=2

acuracy_checking_path = r"modelPoisonDetect\accuracy_checking.pk1"
weigth_update_statistics_path = r"modelPoisonDetect\Weight_update_statistics.pk1"


#@title Example form fields
#@markdown Forms support many types of fields.
dir = False  #@param {type: "boolean"}
make_test = False #@param {type: "boolean"}
shuffle_line = False  #@param {type: "boolean"}
percentage_of_line = 1 #@param {type:"slider", min:0, max:1, step:0.01}
path = r"C:\Users\Giuli\Documents\tesi\federeted\dataset\book"  #@param ['/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset/en/en_US.twitter.txt', '/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset/The_Adventures_of_Sherlock_Holmes.txt', '/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset', 'thursday']

sequence_path=r'globalVariable\sequences.pk1'
vocab_path=r'globalVariable\token.pk1'



emending_length=50  #@param {type: "integer"}

lengt_sequence=3 #@param {type:"slider", min:1, max:10, step:1}
NUM_CLIENTS = 10 #@param {type:"slider", min:0, max:200, step:1}



@tf.autograph.experimental.do_not_convert
def perplexity2(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!

    BTW doesn't really work.
    """
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.pow (2.0, cross_entropy)
    return perplexity

class sparse_recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = tf.one_hot(y_true,  y_pred.shape[1],  dtype='int32')
        super().update_state(y_t,y_pred)

@tf.autograph.experimental.do_not_convert
def crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred)


class attention(Layer):
    def init(self):
        super(attention,self).__init__()
    def build(self,input_shape):
        self.W=self.add_weight(name='att_weight',shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name='att_bias',shape=(input_shape[-2],1),initializer="zeros")        
        super(attention, self).build(input_shape)
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def convert_int_text(prev,next,vocab_inv):
  r = ''
  for i in prev:
    r = r + vocab_inv[i] + " "
  r = r + vocab_inv[next]
  return r