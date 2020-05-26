from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np


cnn = load_model('./models/modelo.h5')
cnn.load_weights('./models/pesos.h5')

def predecir(file):
    x = load_img(file, target_size=(150, 150))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)

    if answer == 0:
        print('Gato')
    elif answer == 1:
        print('Perro')
    elif answer == 2:
        print('Gorila')

    return answer