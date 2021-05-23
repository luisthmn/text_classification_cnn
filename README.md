# Clasificación de texto con CNNs

### Table of contents
* [Descripción](#Información-general)
* [Tecnologias](#Tecnologias)
* [Instrucciones](#Instrucciones-de-uso)

### Información general
Este proyecto consiste en la elaboración de un modelo CNN que pueda clasificar diferentes reviews de películas (En idioma ingles) como positivas o negativas. Las reviews utilizadas para entrenar y probar nuestro modelo son obtenidas de la página [IMDB](https://www.imdb.com/), una base de datos en línea de información relacionada con películas, programas de televisión, videos caseros, videojuegos y contenido de transmisión en línea.

Las reviews utilizadas pueden ser descargadas con este [link](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

### Tecnologias
Este proyecto utiliza la siguiente versión de Keras:
* Keras (2.4.3)
	
### Instrucciones de uso
Para poder utilizar nuestro proyecto, tenemos que descargar las reviews que usaremos, despues se realizará el preprocesamiento del texto, creará nuestro modelo y lo entrenará usando nuestro dataset, al finalizar el proceso se mostrarán los resultados de accuraccy y loss. Todo esto se realiza automáticamente al correr el siguiente archivo:

```
$ python train.py
```

El modelo se guardará dentro del directorio del archivo en una carpeta llamada "model". Puede cargarse nuevamente corriendo la siguiente linea:

```
>>> from tensorflow import keras
>>> model = keras.models.load_model('path/to/location')
```

Para obtener una explicación más detallada de las técnicas de preprocesamiento y el funcionamiento del modelo en el siguiente [artículo de medium](https://luisthmn.medium.com/clasificaci%C3%B3n-de-texto-usando-cnns-526a93ae3828).
