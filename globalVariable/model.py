from globalVariable.global_variable import *


def next_word_model(vocab_size,lengt_sequence,weigth= None,compile=True):
  embedding_matrix = pickle.load(open(r'embended\embended.pk1', 'rb'))

  top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(
      k=3, name='top3', dtype=None
  )
  model = Sequential()
  model.add(Embedding(vocab_size, 50, input_length=lengt_sequence,weights=[embedding_matrix],trainable=True))
  model.add(LSTM(50, return_sequences=True))
  model.add(Attention(units=25))
  model.add(Dropout(0.3))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(vocab_size, activation='softmax'))

  if weigth is not None:
    model.set_weights(weigth)
  if compile:
    optimizer=Adam(learning_rate=0.001) #tf.keras.optimizers.RMSprop()#
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',top_k])
  return model


def distance_weigths_scalar(weigth1, weigth2):
  def euclideanDistance(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y)))
    return dist
  dist=[]
  for i,_ in enumerate(weigth1):
    d=euclideanDistance(weigth1[i], weigth2[i])
    dist.append(d)
  return tf.reduce_sum(dist)/len(dist)
