from global_variable import *
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


lines=lines[:int(len(lines)*percentage_of_line)]
print("count number of line: " + str(len(lines)) )
data = ' '. join(lines)
data=data.lower()
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace("—£","").replace("œuvre","").replace("—","").replace('‘',"").replace('’',"").replace( '£',"").replace('“',"").replace('”',"")

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space

data = data.translate(translator)

data[:360]

tokens = nltk.wordpunct_tokenize(data)

words = [w.lower() for w in tokens]

vocab = sorted(set(words))
vocab_dict = dict() 
vocab_dic_reverse = dict()
for index,value in enumerate(vocab):
  vocab_dict[value] = index
  vocab_dic_reverse[index] = value
  
list(vocab_dict)[-50:]

vocab_size=len(vocab_dict)
print("vocab size",vocab_size)

def convert_int_text(prev,next,vocab_inv):
  r = ''
  for i in prev:
    r = r + vocab_inv[i] + " "
  r = r + vocab_inv[next]
  return r

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

sequences = np.array(sequences)
print("size of sequence", len(sequences))
pickle.dump(sequences, open(r'globalVariable\sequences.pk1', 'wb'))

s = pickle.load(open(r'globalVariable\sequences.pk1', 'rb'))


pickle.dump(vocab_dict, open(r'globalVariable\token.pk1', 'wb'))
vocab =  pickle.load(open(r'globalVariable\token.pk1', 'rb'))


embeddings_index = dict()
f = open(r"C:\Users\Giuli\Documents\tesi\federeted\embended\glove.6B.50d.txt", encoding="utf8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, emending_length))

for i, word in enumerate(list(vocab_dict)):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

pickle.dump(embedding_matrix, open(r'embended\embended.pk1', 'wb'))
