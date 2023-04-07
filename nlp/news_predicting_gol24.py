'''
https://ermlab.com/en/blog/nlp/polish-sentiment-analysis-using-keras-and-word2vec/
https://github.com/Ermlab/pl-sentiment-analysis/blob/master/Models/predict.py
'''

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.utils import pad_sequences
import pickle


# LOADING DATA
print('Loading data...')

news_path_original = './data/news/articles_gol24.json'
news_path_updated = './data/news/articles_gol24_update.json'

json_df_original = pd.read_json(news_path_original)
json_df_updated = pd.read_json(news_path_updated)

json_df = pd.concat([json_df_original, json_df_updated], ignore_index=True)

json_df.to_json('./data/news/articles_gol24.json',
                force_ascii = False, orient="columns")


# LOADING MODEL
print('Loading model...')

model = load_model('./nlp/models/finalsentimentmodel.h5')

print('Loading word index...')
with open('./nlp/models/finalwordindex.pkl', 'rb') as picklefile:
    word_index = pickle.load(picklefile)
top_words = len(word_index)
tokenizer = Tokenizer(num_words=top_words)
tokenizer.word_index = word_index
print('Found %s unique tokens.' % len(word_index))

model.load_weights('./nlp/models/finalsentimentmodel.h5')

# PREDICTIONS
print('Predicting sentiment...')

sample_text = json_df['content_listed']

test_sequences = tokenizer.texts_to_sequences(sample_text)

x_test = pad_sequences(test_sequences, maxlen=40)

print('x_test shape:', str(x_test.shape))

result = model.predict(x_test)

json_df['rate_pos'] = result[:,1]
json_df['rate_neg'] = result[:,2]

print(json_df.sample(2))
print(json_df['rate_pos'].max())
print(json_df['rate_neg'].max())
print(json_df.shape)

json_df = json_df[json_df['length'] != 0]
json_df.drop(axis=1, columns = ['length', 'content_listed', 'content'], inplace=True)

print(json_df['rate_pos'].max())
print(json_df['rate_neg'].max())
print(json_df[(json_df['rate_pos'] > 0.4) & (json_df['rate_pos'] < 0.6)])
print(json_df.shape)

json_df.to_json('./data/news/articles_gol24_nlp.json',
                force_ascii = False,
                orient="columns")
