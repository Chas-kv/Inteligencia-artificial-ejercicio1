import os

# Verificar si el archivo existe
file_path = 'D:/Inteligencia artifical/taller1/mpl_model.h5'
if os.path.exists(file_path):
    print("El archivo existe.")
else:
    print("El archivo no se encuentra en la ruta especificada.")
