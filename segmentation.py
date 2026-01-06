import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Lire l'image en niveaux de gris
img = cv.imread("objs.jpg", cv.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image introuvable. Vérifie le nom ou le chemin.")

ng = np.array(img)

# Calcul de l'histogramme
hist = cv.calcHist([ng], [0], None, [256], [0, 256])

# Supprimer les valeurs faibles (bruit)
for x in range(256):
    if hist[x] < 1000:
        hist[x] = 0

# Récupérer les niveaux de gris significatifs
liste = []
for x in range(256):
    if hist[x] != 0:
        liste.append(x)

h, l = ng.shape

# Affichage image originale
plt.figure(figsize=(5,5))
plt.imshow(ng, cmap='gray')
plt.title("Image originale")
plt.axis('off')

# Segmentation par niveaux détectés
for k in range(len(liste)):
    seg = np.ones((h, l), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(l):
            if ng[i, j] == liste[k]:
                seg[i, j] = ng[i, j]

    plt.figure(figsize=(5,5))
    plt.imshow(seg, cmap='gray')
    plt.title(f"Région segmentée - niveau {liste[k]}")
    plt.axis('off')

plt.show()
