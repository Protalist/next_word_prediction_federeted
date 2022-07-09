from re import X
import sys

from ray import client
from globalVariable.global_variable import *

def client(num_clients, id_client):
    NUM_CLIENTS=num_clients
    cid=id_client

    vocab_dict =  pickle.load(open(r'globalVariable\token.pk1', 'rb'))
    sequences = pickle.load(open(r'globalVariable\sequences.pk1', 'rb'))
    vocab_size=len(vocab_dict)

    class NextWordPredictionClient(fl.client.NumPyClient):
        def __init__(self, model, x_train, y_train, x_val, y_val) -> None:
            self.model = model
            self.x_train, self.y_train = x_train, y_train
            self.x_val, self.y_val = x_val, y_val

        def get_parameters(self):
            return self.model.get_weights()

        def fit(self, parameters, config):
            """
            self.model.fit(self.x_train,self.y_train, epochs=5, batch_size=64)
            return self.model.get_weights(), len(self.x_train), {}"""

            self.model.set_weights(parameters)
            self.shuffle()
            self.model.fit(self.x_train_m,self.y_train_m, epochs=3, batch_size=64)
            delta=np.array(self.model.get_weights())-np.array(parameters)
            return delta, len(self.x_train_m), {"cid":cid}

        def evaluate(self, parameters, config):
            self.shuffle()
            self.model.set_weights(parameters)
            loss, accuracy,top_3 = self.model.evaluate(self.x_val_m, self.y_val_m)
            return loss, len(self.x_val), {"val_loss": loss, "val_accuracy":accuracy , "val_top_3": top_3}
        
        def shuffle(self):
            X_f= np.concatenate((self.x_train , self.x_val))
            y_f= np.concatenate((self.y_train , self.y_val))
            p = np.random.permutation(len(X_f))
            X_f=X_f[p]
            y_f=y_f[p]
            #rand = random.randint(1, split)
            idx=int(len(X_f)/2)
            X_f = X_f[:idx]
            y_f = y_f[:idx]
            split_idx = math.floor(len(X_f) * 0.9)
            self.x_train_m, self.y_train_m = X_f[:split_idx], y_f[:split_idx]
            self.x_val_m, self.y_val_m = X_f[split_idx:], y_f[split_idx:]


    embedding_matrix = pickle.load(open(r'embended\embended.pk1', 'rb'))

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
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy',top_k])

    partition_size = math.floor(len(sequences) / (NUM_CLIENTS+1))
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size

    X_f = []
    y_f = []

    for i in sequences[idx_from: idx_to]:
        X_f.append(i[0:lengt_sequence])
        y_f.append(i[-1])
    
    if cid=="7":
        idx=len(y_f)
        for index, item in enumerate(y_f):
            if  index>idx*0.3 and index<idx*0.4:
                y_f[index] = 7211
    X_f = np.array(X_f)
    y_f = np.array(y_f)

     
    print("Number of trining element",len(X_f))

    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(X_f) * 0.9)
    x_train, y_train = X_f[:split_idx], y_f[:split_idx]
    x_val, y_val = X_f[split_idx:], y_f[split_idx:]


    param=model.get_weights()
   # NextWordPredictionClient( model, x_train, y_train, x_val, y_val).fit(param,None)
    fl.client.start_numpy_client("localhost:3031", client=NextWordPredictionClient( model, x_train, y_train, x_val, y_val))
    #text = input("Enter your line: ")

if __name__ == "__main__":
    print (sys.argv)
    b=0
    c=0
    try:
        b = int(sys.argv[1])
        c = sys.argv[2]
    except IndexError:
        b = 10
        c = 1
    import random
    import time
    timr = random.randint(1,1)
    time.sleep(timr)
    client(b,c)