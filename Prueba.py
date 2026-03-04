import os
import re
import matplotlib.pyplot as plt
from collections import Counter

# %matplotlib inline  <-- (Eliminado o comentado, solo funciona en Jupyter Notebook)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ya no es necesario os.getcwd() si vas a usar una ruta absoluta
imgpath = '/Users/gabriels/Proyectos/Samsung/Dataset2/archive/'

images = []
dircount = Counter()  # Diccionario optimizado para contar
cant_total = 0

print("Leyendo imágenes de:", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        # Se agrega re.IGNORECASE para coincidir con .JPG o .jpg en mayúsculas/minúsculas
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename, re.IGNORECASE):
            cant_total += 1
            filepath = os.path.join(root, filename)
            
            # Cargar imagen en memoria (¡precaución si son muchas/muy pesadas!)
            image = plt.imread(filepath)
            images.append(image)
            
            # Registramos 1 imagen sumada al conteo del directorio actual
            dircount[root] += 1
            
            print(f"Leyendo... {cant_total}", end="\r")

print("\n\n--- Resultados de lectura ---")
print(f"Directorios leídos con imágenes: {len(dircount)}")
print("Imágenes en cada directorio:")
for directorio, cantidad in dircount.items():
    print(f"  - {directorio}: {cantidad}")
print(f"Suma total de imágenes en subdirs: {cant_total}")
