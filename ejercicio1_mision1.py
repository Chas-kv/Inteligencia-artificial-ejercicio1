import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.datasets import mnist

import numpy as np

import matplotlib.pyplot as plt
import os

from PIL import Image
# Paso 2: Cargar el conjunto de datos MNIST

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Paso 3: Preprocesar los datos

x_train = x_train.astype('float32') / 255  # Normalización

x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28 * 28)     # Aplanar las imágenes

x_test = x_test.reshape(-1, 28 * 28)
y_train = tf.keras.utils.to_categorical(y_train, 10)  # One-hot encoding

y_test = tf.keras.utils.to_categorical(y_test, 10)
# Paso 4: Definir el modelo MLP

model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))  # 10 clases de salida
# Paso 5: Compilar el modelo

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

# Paso 6: Entrenar el modelo

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Paso 7: Evaluar el modelo

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Precisión en el conjunto de prueba: {test_acc}')

# Paso 8: Probar el modelo con una imagen del conjunto de prueba

imagen = x_test[0].reshape(1, 28 * 28)  # Seleccionar la primera imagen

prediccion = model.predict(imagen)

digit_predicho = np.argmax(prediccion)

# Mostrar la imagen y el dígito predicho

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')

plt.title(f'Predicción: {digit_predicho}')

plt.show()

# Paso 9: Probar el modelo con una imagen personalizada

#ruta_imagen="D:/talento tech/ia/imagen1.png"


def predecir_imagen_personalizada(ruta_imagen):
    try:
        print(f"Procesando imagen: {ruta_imagen}")  # Debug
        img = Image.open(ruta_imagen).convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))  # Redimensionar a 28x28 píxeles
        img_array = np.array(img).reshape(1, 28 * 28).astype('float32') / 255  # Normalizar

        print(f"Imagen procesada correctamente: {ruta_imagen}")  # Debug

        prediccion = model.predict(img_array)  # Hacer la predicción
        digit_predicho = np.argmax(prediccion)

        print(f'Predicción para {os.path.basename(ruta_imagen)}: {digit_predicho}')  # Imprimir resultado

        plt.imshow(img, cmap='gray')
        plt.title(f'Predicción: {digit_predicho}')
        plt.show()
    
    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {e}")

# Función para iterar sobre todas las imágenes en una carpeta
def predecir_imagenes_en_carpeta(ruta_carpeta):
    print(f"Procesando imágenes en la carpeta: {ruta_carpeta}")
    
    archivos_encontrados = os.listdir(ruta_carpeta)  # Obtener la lista de archivos
    
    if len(archivos_encontrados) == 0:
        print(f"No se encontraron archivos en la carpeta: {ruta_carpeta}")
    
    for archivo in archivos_encontrados:
        print(f"Archivo encontrado: {archivo}")  # Imprimir cada archivo encontrado
        
        if archivo.endswith(('.png', '.jpg', '.jpeg')):  # Verificar si es una imagen
            ruta_imagen = os.path.join(ruta_carpeta, archivo)
            predecir_imagen_personalizada(ruta_imagen)
        else:
            print(f"{archivo} no es una imagen válida.")  # Archivos no válidos

# Ruta de la carpeta de imágenes de prueba
ruta_carpeta = 'D:/Inteligencia artifical/test2/'

# Ejecutar la predicción para todas las imágenes en la carpeta
predecir_imagenes_en_carpeta(ruta_carpeta)
