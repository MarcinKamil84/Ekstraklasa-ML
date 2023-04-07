'''
https://ermlab.com/en/blog/nlp/polish-sentiment-analysis-using-keras-and-word2vec/
https://github.com/Ermlab/pl-sentiment-analysis/blob/master/Models/predict.py
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
np.random.seed(7)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.layers import Embedding
from keras.utils import np_utils, pad_sequences
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors, word2vec
import gensim
from gensim.utils import simple_preprocess
from keras.utils import to_categorical
import pickle
import h5py
from time import time


# IMPORTING SENTIMENT DATASET FOR TRAINING

filename = './nlp/data/polish_sentiment_dataset.csv'

dataset = pd.read_csv(filename, delimiter = ",")

# Delete unused column
del dataset['length']

# Delete All NaN values from columns=['description','rate']
dataset = dataset[dataset['description'].notnull() & dataset['rate'].notnull()]

# We set all strings as lower case letters
dataset['description'] = dataset['description'].str.lower()


# SPLITTING DATA FOR TRAINING

X = dataset['description']
y = dataset['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
print("X_val shape: " + str(X_val.shape))
print("y_train shape: " + str(y_train.shape))
print("y_test shape: " + str(y_test.shape))
print("y_val shape: " + str(y_val.shape))


# LOADING EXISTING WORD2VEC MODEL

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('nkjp-forms-all-100-cbow-hs.txt', binary=False)

embedding_matrix = word2vec_model.wv.syn0
print('Shape of embedding matrix: ', embedding_matrix.shape)


# VECTORIZING

top_words = embedding_matrix.shape[0]
mxlen = 50
nb_classes = 3

tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequences_train, maxlen=mxlen)
X_test = pad_sequences(sequences_test, maxlen=mxlen)
X_val = pad_sequences(sequences_val, maxlen=mxlen)

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
y_val = np_utils.to_categorical(y_val, nb_classes)


# LONG SHORT-TERM MEMORY CONFIGURATION

batch_size = 32
nb_epoch = 12

embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, recurrent_dropout=0.5, dropout=0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()


# COMPILING THE MODEL

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
rnn = model.fit(X_train, y_train, epochs= nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_val, y_val))
score = model.evaluate(X_val, y_val)
print("Test Loss: %.2f%%" % (score[0]*100))
print("Test Accuracy: %.2f%%" % (score[1]*100))


# SAVING MODEL

print('Save model...')
model.save('./nlp/models/finalsentimentmodel.h5')
print('Saved model to disk...')

print('Save Word index...')
output = open('models/finalwordindex.pkl', 'wb')
pickle.dump(word_index, output)
print('Saved word index to disk...')


# VISUALIZATIONS

plt.figure(0)
plt.plot(rnn.history['accuracy'],'r')
plt.plot(rnn.history['val_accuracy'],'g')
plt.xticks(np.arange(0, nb_epoch+1, nb_epoch/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy LSTM l=10, epochs=20") # for max length = 10 and 20 epochs
plt.legend(['train', 'validation'])
plt.savefig("figure1.jpg")

plt.figure(1)
plt.plot(rnn.history['loss'],'r')
plt.plot(rnn.history['val_loss'],'g')
plt.xticks(np.arange(0, nb_epoch+1, nb_epoch/5))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Training vs Validation Loss LSTM l=10, epochs=20") # for max length = 10 and 20 epochs
plt.legend(['train', 'validation'])
plt.show()
plt.savefig("figure2.jpg")