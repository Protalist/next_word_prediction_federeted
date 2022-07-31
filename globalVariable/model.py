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
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=['accuracy',top_k])
  return model


def distance_weigths_scalar(weigth1, weigth2):
  def euclideanDistance(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.pow(x - y)))
    return dist
  dist=[]
  for i,_ in enumerate(weigth1):
    d=euclideanDistance(weigth1[i], weigth2[i])
    dist.append(d)
  return tf.reduce_sum(dist)/len(dist)

def distance_lp_norm(weigth1, weigth2):
  distance = np.array(weigth1)-np.array(weigth2)
  d=[]
  for layer in distance:
    for neuron in layer:
      if not (type(neuron) == np.ndarray):
        d.append(pow(neuron,2))
        break
      for weight in neuron:
        d.append(pow(weight,2))
  return np.sqrt(np.sum(d))

@tf.function
def train_step(dataset, model,loss=None,e1=0):
  with tf.GradientTape() as tape:
    predictions = model(dataset[0], training= True)
    if loss is None:
      loss = model.loss(dataset[1], predictions)
    else: 
      loss = loss(dataset[1], predictions)+e1
    gradients = tape.gradient(loss, model.trainable_variables)

  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def train(dataset, epochs, batch_size, model,loss=None):
  l,e1=loss(model.get_weights())
  for epoch in range(epochs):
    print(epoch)
    loss_list = []
    i=0
    data = dataset[0]
    label = dataset[1]
    while i < len(data):
      t = train_step([data[i:i+batch_size],label[i:i+batch_size]], model,l,e1)
      loss_list.append(t)
      i=i+batch_size

    loss = sum(loss_list) / len(loss_list)
    print (f'Epoch {epoch+1}, loss={loss}')