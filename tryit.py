import pickle
from keras_preprocessing import sequence
import sys
from tensorflow.keras.models import load_model


def clasificaReview(review):

    pad_length = 1000
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    model = load_model('model')

    prediccion = model.predict_classes([sequence.pad_sequences(
        tokenizer.texts_to_sequences([review]), maxlen=pad_length, padding='post')])[0][0]

    if(prediccion):
        print("La review es positiva")
        return True
    else:
        print("La review es negativa")
        return False


print("Escribe la review de prueba: ")
review = input()
clasificaReview(review)
