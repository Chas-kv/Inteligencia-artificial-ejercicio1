# Proyecto: Modelo de Inteligencia Artificial para Predicción de Dígitos MNIST

Este repositorio contiene ejemplos en Python de un modelo de inteligencia artificial basado en el conjunto de datos MNIST. El modelo está diseñado para predecir números escritos a mano (0-9). Sin embargo, su precisión es alta solo cuando se le proporcionan imágenes similares a las utilizadas durante su entrenamiento.

## Archivos del Proyecto

### Archivos principales
- **`Ejercicio_mision1.ipynb`**: Notebook en el que se observa el modelo en acción.
- **`app.py`**: Aplicación basada en Streamlit para probar el modelo.
- **`crearmodelo.ipynb`**: Notebook utilizado para la creación del modelo y generación del archivo `.h5`.
- **`crearmodelo.py`**: Script Python que también permite la creación del modelo MNIST.
- **`ejercicio1_mision1.py`**: Código en Python para probar el modelo con datos MNIST.
- **`mpl_model.h5`**: Archivo del modelo entrenado en formato HDF5.

### Archivos adicionales
- **`comandos.txt`**: Archivo con comandos utilizados durante el desarrollo del proyecto.
- **`requeriments.txt`**: Lista de dependencias necesarias para ejecutar el código.

## Instalación y Ejecución

### 1. Clonar el repositorio
```bash
 git clone https://github.com/tu_usuario/tu_repositorio.git
 cd tu_repositorio
```

### 2. Instalar dependencias
Se recomienda utilizar un entorno virtual para la instalación de las dependencias:
```bash
pip install -r requeriments.txt
```

### 3. Ejecutar la aplicación con Streamlit
Para probar el modelo en la interfaz de usuario de Streamlit, ejecutar:
```bash
streamlit run app.py
```
Esto abrirá la interfaz en el navegador donde se podrán cargar imágenes y observar las predicciones del modelo.

## Notas
- El modelo predice correctamente los dígitos cuando las imágenes son similares a las del entrenamiento.
- Si se utilizan imágenes con estilos diferentes, la precisión puede verse afectada.
- Para mejorar el modelo, es recomendable entrenarlo con un conjunto de datos más variado o aplicar técnicas de data augmentation.

## Autor
_Chas-kv_

---
Cualquier mejora o sugerencia es bienvenida. ¡Gracias por visitar este repositorio! 🚀

