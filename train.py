import tokenizer
import reviews
import stopwords

import tempfile
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.python.keras.optimizers import TFOptimizer
import pickle


# Iniciamos descargando las revies
reviews.download_reviews()


# Parametros de nuestro modelo y nuestro tokenizer
vocab_size = 2000       # Cantidad de palabras que estaremos manejando
maxlen = 1000           # Longitud máxima de una review
batch_size = 32        # 32
embedding_dims = 50    # Cantidad de dimensiones en el Embedding Layer    10
filters = 16           # 16
kernel_size = 3        # 3
hidden_dimensions = 250
epochs = 15           # 25

# Cargamos nuestro dataset con las reviews
(X_train, y_train), (X_test, y_test) = reviews.load_imdb()

# Pasamos todas las reviews a minusculas
X_train = map(str.lower, X_train)
X_test = map(str.lower, X_test)

# Cargamos nuestras stopwords
stopwords_list = stopwords.load_stopwords()
print("Stopwords loaded")

# Eliminamos las stopwords de nuestro dataset
X_train = stopwords.remove_stopwords(X_train, stopwords_list)
X_test = stopwords.remove_stopwords(X_test, stopwords_list)
print("Stopwords removed from dataset")

# Creamos nuestro tokenizer, lo entrenamos con nuestro dataset y lo guardamos
tokenizer.train_and_save_tokenizer(X_train, vocab_size)
print("Tokenizer trained and stored")

# Cargamos nuestro tokenizer que acabamos de guardar
loaded_tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
print("Tokenizer has been loaded")

# Tokenizamos el texto
X_train = tokenizer.tokenize_text(X_train, loaded_tokenizer, maxlen)
X_test = tokenizer.tokenize_text(X_test, loaded_tokenizer, maxlen)
print("Dataset Tokenized")


# Creamos nuestro modelo CNN para clasificacion de texto
model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dims,
                    input_length=maxlen,
                    mask_zero=True
                    ))
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.5))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(hidden_dimensions, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Compilamos nuestro modelo
model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',  # adam
              metrics=['accuracy'])

# Entramos nuestro modelo
es_callback = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          # callbacks=[es_callback]
          )

# Imprimimos los resultados
results = model.evaluate(X_test[1000:], y_test[1000:])
print("test loss: %.2f" % results[0])
print("test accuracy: %.2f%%" % (results[1] * 100))

# Guardamos el modelo

model.save('model')
