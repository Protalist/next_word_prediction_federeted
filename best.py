# %% [markdown]
# #Utilities

# %%
from globalVariable.global_variable import *

# %% [markdown]
# ## Import

# %%
"""import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM,GRU, Dense,Bidirectional,Dropout,Activation, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K

import string
import pickle
import numpy as np
import os
import math

import matplotlib.pyplot as plt
import nltk

from attention import Attention
import string
"""

# %%
#@title Install flower simulation
#!pip install flwr["simulation"]==0.18.0 

# %% [markdown]
# ## Global Variable 

# %%
#@title Example form fields
#@markdown Forms support many types of fields.
dir = True  #@param {type: "boolean"}
make_test = False #@param {type: "boolean"}
shuffle_line = True  #@param {type: "boolean"}
percentage_of_line = 1 #@param {type:"slider", min:0, max:1, step:0.01}
path = r"dataset\en_US\short\short_tweet.txt"  #r"C:\Users\Giuli\Documents\tesi\federeted\dataset\book"  #@param ['/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset/en/en_US.twitter.txt', '/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset/The_Adventures_of_Sherlock_Holmes.txt', '/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset', 'thursday']
token_path="/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/tokenization/tokenizer1.pkl"
checkpoint_path="/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/saved_model/nextword1.h5"

emending_length=50  #@param {type: "integer"}

lengt_sequence=3 #@param {type:"slider", min:1, max:10, step:1}
NUM_CLIENTS = 162 #@param {type:"slider", min:0, max:200, step:1}

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %% [markdown]
# ## Function

# %%
@tf.autograph.experimental.do_not_convert
def perplexity(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!

    BTW doesn't really work.
    """
    cross_entropy = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
    perplexity = tf.keras.backend.pow(2.0, cross_entropy)
    return perplexity

@tf.autograph.experimental.do_not_convert
def crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred)

class attentions(Layer):
    def init(self):
        super(attentions,self).__init__()
    def build(self,input_shape):
        self.W=self.add_weight(name='att_weight',shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name='att_bias',shape=(input_shape[-2],1),initializer="zeros")        
        super(attentions, self).build(input_shape)
    def call(self,x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Load File

# %%
lines=[]
if dir:
  with open(path, 'r', encoding = "utf8") as file:
    print(file)
    for i in file:
      lines.append(i)
    print(len(lines))
else:
  for filename in os.listdir(path):
    if filename.endswith(".txt"):
      print(filename)
      with open(os.path.join(path, filename), 'r', encoding = "utf8") as file:
        print(file)
        for i in file:
          lines.append(i)
        print(len(lines))

# %%
if shuffle_line:
  import random
  random.shuffle(lines)

# %% [markdown]
# ### Print example

# %%
print("The First Line: ", lines[1])
print("The Last Line: ", lines[-1])

# %% [markdown]
# ## Tokenization

# %%
lines=lines[:int(len(lines)*percentage_of_line)]
print("count number of line: " + str(len(lines)) )
data = ' '. join(lines)
data=data.lower()
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace("—£","").replace("œuvre","").replace("—","").replace('‘',"").replace('’',"").replace( '£',"").replace('“',"").replace('”',"")

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

data = data.translate(translator)

data[:360]

# %%
tokens = nltk.wordpunct_tokenize(data)

# %%
tokens[:15]

# %%
words = [w.lower() for w in tokens]

# %%
words[:15]

# %%
vocab = sorted(set(words))
vocab_dict = dict() 
vocab_dic_reverse = dict()
for index,value in enumerate(vocab):
  vocab_dict[value] = index
  vocab_dic_reverse[index] = value
  
list(vocab_dict)[-50:]

# %%
vocab_dic_reverse[500] 

# %%
vocab_size=len(vocab_dict)
vocab_size

# %%
def convert_int_text(prev,next,vocab_inv):
  r = ''
  for i in prev:
    r = r + vocab_inv[i] + " "
  r = r + vocab_inv[next]
  return r

# %% [markdown]
# ## Create sequence

# %%
X=[]
y=[]
sequences=[]
for i in range(len(words)-lengt_sequence):
  sequence = []
  for j in range(lengt_sequence+1):
    sequence.append(vocab_dict[words[i+j]])
  sequences.append(sequence)

np.random.shuffle(sequences)
for sequence in sequences:
  X.append(sequence[:-1])
  y.append(sequence[-1])
X = np.array(X)
y = np.array(y)

# %%
len(sequences)

# %%
print(X[0], end=' => ')
print(y[0])
print(convert_int_text(X[0],y[0],vocab_dic_reverse))

# %% [markdown]
# # Model

# %%
embeddings_index = dict()
f = open(r"C:\Users\Giuli\Documents\tesi\federeted\embended\glove.6B.50d.txt", encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# %%

embedding_matrix = np.zeros((vocab_size, emending_length))

for i, word in enumerate(list(vocab_dict)):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# %% [markdown]
# ## Sequential

# %%
model = Sequential()
model.add(Embedding(vocab_size, emending_length, input_length=lengt_sequence,weights=[embedding_matrix],trainable=True))
model.add(LSTM(50, return_sequences=True))
model.add(Attention(units=25))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))


# %%
model.summary()

# %% [markdown]
# ## callbacks

# %%
checkpoint_path="/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/saved_model/nextword1.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto')

reduce_l = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
#tensorboard_Visualization = TensorBoard(log_dir=logdir)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_top3', patience=7, verbose=1,  mode="max")
top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k=3, name='top3', dtype=None
)

# %%
@tf.autograph.experimental.do_not_convert
def pp(y_true, y_pred):
    """
    The perplexity metric. Why isn't this part of Keras yet?!

    BTW doesn't really work.
    """
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.pow(tf.constant([2.0]),-1*cross_entropy)
    return perplexity

# %%
optimizer=Adam(learning_rate=0.001) #tf.keras.optimizers.RMSprop()#
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',top_k,pp])

# %%
batch_size= pow(2,8)
print(batch_size)
history = model.fit(X, y, validation_split=0.03, epochs=5, batch_size=batch_size,callbacks=[early_stop,reduce_l],shuffle=True)

# %% [markdown]
# ## plot the result
# 

# %%
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20,10))
ax[0].plot(history.history['loss'])
ax[1].plot(history.history['top3'])
ax[2].plot(history.history['accuracy'])
ax[0].plot(history.history['val_loss'])
ax[1].plot(history.history['val_top3'])
ax[2].plot(history.history['val_accuracy'])
ax[0].set_title('model loss')
ax[1].set_title('model top3')
ax[1].set_title('model accuracy')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[1].set_ylabel('top3')
ax[1].set_xlabel('epoch')
ax[2].set_ylabel('accuracy')
ax[2].set_xlabel('epoch')
fig.tight_layout(pad=10.0)
plt.show()

# %% [markdown]
# #test

# %%
# Importing the Libraries


# Load the model and tokenizer
#checkpoint_path="/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/saved_model/nextword1.h5"

"""model = load_model(checkpoint_path,custom_objects={"crossentropy": crossentropy, "perplexity":perplexity})
model.summary
tokenizer = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/tesi/next_word_prediction/dataset/tokenization/tokenizer1.pkl", 'rb'))
"""

# %%
def Predict_Next_3_Words(model, tokenizer, text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the the predicted word.
    
    """
    sequence = tokenizer.texts_to_sequences([text])
    sequence[0] = sequence[0][-lengt_sequence:]
    if len(sequence[0]) < lengt_sequence:
      return "no resutl"
    sequence = sequence[-lengt_sequence:]

    for i in range(3):
      print("i am thinking . . .")
      preds = model.predict(sequence)
      predicted_word=""
      pred_l=preds[0]
      ind = np.argpartition(pred_l, -3)[-3:]
      pred_l = ind[np.argsort(pred_l[ind])]
      pred_l = np.flip(pred_l, axis=None)
      result = []
      for ind in pred_l:
        print(preds[0][ind])
        predict = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(ind)]
        result.append(predict)
      return result

# %%
"""text1 = "i love to"
text = text1.split(" ")
sequence = tokenizer.texts_to_sequences([text])
sequence[0] = sequence[0][-lengt_sequence:]
sequence = sequence[-lengt_sequence:]
preds = model.predict(sequence)
pred_l=preds[0]
ind = np.argpartition(pred_l, -3)[-3:]
print(ind)
print(pred_l[1])
print(ind[np.argsort(pred_l[ind])])
pred_l = ind[np.argsort(pred_l[ind])]
pred_l = np.flip(pred_l, axis=None)
result = []

for ind in pred_l:
  print(preds[0][ind])
  predict = list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(ind)]
  result.append(predict)
print(result)
p=Predict_Next_3_Words(model, tokenizer, text)"""

# %%

"""
while(make_test):

    text = input("Enter your line: ")
    
    if text == "stop":
        print("Ending The Program.....")
        break
    
    else:
        try:
            text = text.split(" ")
            p=Predict_Next_3_Words(model, tokenizer, text)
            print(p)
        except NameError:
            print(NameError)
            continue"""

# %% [markdown]
# # Fedeteretd simulate

# %% [markdown]
# ## Import

# %%

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# %% [markdown]
# # Create client fr flower 

# %%
class NextWordPredictionClient(fl.client.NumPyClient):
    def __init__(self, model_f, x_train, y_train, x_val, y_val) -> None:
      self.model_f = model_f
      self.x_train, self.y_train = x_train, y_train
      self.x_val, self.y_val = x_val, y_val

    def get_parameters(self):
        return self.model_f.get_weights()

    def fit(self, parameters, config):
        self.model_f.set_weights(parameters)
        self.model_f.fit(self.x_train,self.y_train, epochs=5, batch_size=64,verbose=2)
        return self.model_f.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        self.model_f.set_weights(parameters)
        loss, accuracy,top_3 = self.model_f.evaluate(self.x_val, self.y_val,verbose=2)
        return loss, len(self.x_val), {"val_loss": loss, "val_accuracy":accuracy , "val_top_3": top_3}


# %% [markdown]
# ## Implementation

# %%
np.random.shuffle(sequences)

# %%
def client_fn(cid: str) -> fl.client.Client:
  # Create model

  top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k=3, name='top3', dtype=None
  )
  model = Sequential()
  model.add(Embedding(vocab_size, emending_length, input_length=lengt_sequence,weights=[embedding_matrix],trainable=True))
  model.add(LSTM(50, return_sequences=True))
  model.add(Attention(units=25))
  model.add(Dropout(0.3))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(vocab_size, activation='softmax'))
  
  optimizer=Adam(learning_rate=0.001) #tf.keras.optimizers.RMSprop()#
  model.compile(loss=perplexity, optimizer=optimizer, metrics=['accuracy',top_k])
  
  partition_size = math.floor(len(sequences) / (NUM_CLIENTS+1))
  idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

  X_f = []
  y_f = []

  for i in sequences[idx_from: idx_to]:
      X_f.append(i[0:lengt_sequence])
      y_f.append(i[-1])
      
  X_f = np.array(X_f)
  y_f = np.array(y_f)


  # Use 10% of the client's training data for validation
  split_idx = math.floor(len(X_f) * 0.9)
  x_train_cid, y_train_cid = X_f[:split_idx], y_f[:split_idx]
  x_val_cid, y_val_cid = X_f[split_idx:], y_f[split_idx:]
  # Create and return client
  return NextWordPredictionClient(model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

# %%
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    def get_data():
      partition_size = math.floor(len(sequences) / (NUM_CLIENTS+1))
      idx_from, idx_to =  (NUM_CLIENTS+1) * partition_size, (NUM_CLIENTS+2) * partition_size

      X_f = []
      y_f = []

      for i in sequences[idx_from: idx_to]:
          X_f.append(i[0:lengt_sequence])
          y_f.append(i[-1])
          
      X_f = np.array(X_f)
      y_f = np.array(y_f)
      return X_f,y_f

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights):
        model.set_weights(weights)  # Update model with the latest parameters
        x_val, y_val = get_data()
        loss, accuracy,top_3 = model.evaluate(x_val, y_val,verbose=2)
        return loss, { "loss":loss,"val_accuracy":accuracy , "val_top_3": top_3}


    return evaluate

# %%
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results,
        failures,
    ) :
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None
        
        loss,_ = super().aggregate_evaluate(rnd, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["val_accuracy"] * r.num_examples for _, r in results]
        top_k = [r.metrics["val_top_3"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        top_k_grragated = sum(top_k) / sum(examples)
        print(f"Round {rnd} aggregated  from client results loss : {loss},accuracy : {accuracy_aggregated} , top_3: {top_k_grragated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return loss, {"accuracy":accuracy_aggregated,"top_k":top_k_grragated}

# %%
#initial_parameters=model.get_weights()

strategy=AggregateCustomMetricStrategy(
        fraction_fit=0.05,  # Sample 10% of available clients for training
        fraction_eval=0.02,  # Sample 5% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 10 clients for training
        min_eval_clients=2,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
        #initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
        #fit_metrics_aggregation_fn =fit_metrics_aggregation_fn_custom
        #eval_fn=get_eval_fn(model)
)

# %%
import ray
history_fede=fl.simulation.start_simulation(
  client_fn=client_fn,
  num_clients=NUM_CLIENTS,
  strategy=strategy,
  num_rounds=40,
  ray_init_args={})

# %%
ray.init()

# %% [markdown]
# 

# %%
fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(20,10))
ax[0].plot(*zip(*history_fede.losses_distributed))
ax[1].plot(*zip(*history_fede.metrics_distributed['top_k']))
ax[2].plot(*zip(*history_fede.metrics_distributed['accuracy']))
ax[0].set_title('model loss')
ax[1].set_title('model top3')
ax[2].set_title('model accuracy')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('round')
ax[1].set_ylabel('top3')
ax[1].set_xlabel('round')
ax[2].set_ylabel('accuracy')
ax[2].set_xlabel('round')
fig.tight_layout(pad=10.0)
plt.show()


