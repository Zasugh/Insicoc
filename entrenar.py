from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

import sys
import os

#Limpiamos sesiones existentes
K.clear_session()

#Guardamos cargamod los path de los archivo para entrenar la neurona
data_train = './data/entrenamiento'
data_validation = './data/validacion'

'''
Declaramos variables para el funcionamiento de la neurona
    * Ciclos de aprendizaje                     ---> epocas
    * largo de imagenes                         ---> longitud
    * alto de imagenes                          ---> altura
    * tamaño de valos a utilizar en un lote     ---> batch_size
    * pesos de validaciones                     ---> validation_steps
    * redefinición de tamaño de img             ---> filtrosConv1/filtrosConv2
    * tamaño de matriz a redimencionar          ---> tamano_filtro1/tamano_filtro2
    * tamaño de agrupación                      ---> tamano_pool
    * numero de objetos a aprender              ---> clases
    * error en semejanza                        ---> lr
'''
epocas = 20
longitud = 150
altura = 150
batch_size = 32
pasos = 1000
validation_steps = 300
filtrosConv1 = 32
filtrosConv2 = 62
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0004


'''
Entrenamiento de imagen.
    * indicamos una escala superior a la origina
    * indicamos que debe crecer la imagen para analizar
    * verificar la imagen de manera horizontal
'''
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


'''
Entrenamiento del modelo respecto a los arhivos cargados
'''
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_train,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')


'''
Validación de los resultados en el entrenamiento anterior
'''
validacion_generador = test_datagen.flow_from_directory(
    data_validation,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

'''
Tras la validación de agregan capas en 2D para verificar en 3 dimensiones
utilizamos lo definido en las variables iniciales
    * filtrosConv1
    * tamano_filtro1
agregamos una capa de densidad
'''
cnn = Sequential()
cnn.add(
    Convolution2D(
        filtrosConv1,
        tamano_filtro1,
        padding="same",
        input_shape=(longitud, altura, 3),
        activation='relu'
    )
)
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))


'''
Compilamos todos lo trabajado con el fin de preparar el entrenamiento de la
neurona.
'''
cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=lr),
    metrics=['accuracy']
)

'''
Entrenamos la neurona, indicando el periodo de entrenamiento y agregando los
valores obtenidos en, en generador y la validación
'''
cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps
)

'''
Almacenamos lo relizado en el entrenamiento dentro de un modelo
con el fin de poder usar la red en cualquier momento
'''
target_dir = './modelo/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')