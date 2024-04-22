# Ángel Isaac Gómez Canales
# Descriptores 16/04/2023
# Codigo basado en: https://github.com/Jegovila/cursoVR/blob/main/Practica9%3A%20Descriptores/python/practica9.py

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def capturarImgs(cap):
    """
    Esta funcion se usa para capturar las dos imagenes para generar una imagen panoramica
    - Input: cap: webcam
    - Output: Img panoramicaq
    """
    # Tomar imagen 1
    while True:
        # Mostrar webcam
        _, I1 = cap.read()
        cv.imshow("Imagen 1(presionar q para capturar imagen)", I1)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    # Tomar imagen 2
    while True:
        # Mostrar webcam
        _, I2 = cap.read()
        cv.imshow("Imagen 2(presionar q para capturar imagen)", I2)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    return I1, I2

# Configurar webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Tomar las dos imagenes a unir
I1, I2 = capturarImgs(cap)

# Crear descriptor y matcher
akaze = cv.AKAZE_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Calcular Keypoints y Features de cada imagen
keypoints1, features1 = akaze.detectAndCompute(I1, None)
keypoints2, features2 = akaze.detectAndCompute(I2, None)

# Encontrar matches entre las features y ordenarlos
matches = bf.match(features1, features2)
matches = sorted(matches, key=lambda x: x.distance)

# Obtener los top 200 matches para calcular matriz de homografia
left = []
right = []
for m in matches[:100]:
    l = keypoints1[m.queryIdx].pt
    r = keypoints2[m.trainIdx].pt
    left.append(l)
    right.append(r)

# Obtener matriz de homografia
M, _ = cv.findHomography(np.float32(right), np.float32(left))

# Obtener tamaño de nueva imagen
dims = (I1.shape[1] + I2.shape[1], max(I1.shape[0]+100, I2.shape[0]+100))

# Transformar la imagen 2 a la misma perspectiva que la 1
comb = cv.warpPerspective(I2, M, dims)

# Combinar las imagenes
comb[:I1.shape[0], :I1.shape[1]] = I1

# Recortar imagen para eliminar bordes negros
r_crop = 1000  # Recortar hasta donde aparezca el primer pixel negro
comb = comb[:, :r_crop]

cv.imshow("Matches", cv.drawMatches(I1, keypoints1, I2, keypoints2, matches[:100], None, matchColor=(0,0,255), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
cv.imshow("Imagen Combinada", comb)
cv.waitKey(0)


