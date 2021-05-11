import numpy as np
from keras.preprocessing import text
from keras.preprocessing import sequence
import pickle
import pathlib


def train_and_save_tokenizer(corpus, vocab_size):
    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(corpus)
    pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))


def tokenize_text(corpus, tokenizer, pad_length=0):
    corpus = tokenizer.texts_to_sequences(corpus)

    if(pad_length):
        corpus = sequence.pad_sequences(
            corpus, maxlen=pad_length, padding='post')

    corpus = np.array(corpus, dtype=np.int32)

    return corpus
