# Importar librerias requeridas
import cv2
import numpy as np
import math

"""
Optimizaciones:
1. Cambiar el shape de dst de (m, n, 1) a (m, n)

"""


# Capturar video de la webcam
cap = cv2.VideoCapture(0)
# Ajustar el tamano de la figura donde se mostrara el video capturado
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Tomar captura para calcular el tamaño y generar imagen destino
# el primer valor devuelto por read es un booleano que indica si se capturo o no el fotograma
_, src = cap.read()
dimensiones = src.shape

# Funciones callback necesarias para el trackbar
def rot(x):
    pass
def trans(x):
    pass

# Crear la figura donde se mostrara la imagen transformada y el trackbar
cv2.namedWindow("Imagen Transformada")
cv2.createTrackbar("theta", "Imagen Transformada", 0, 360, rot)
cv2.createTrackbar("t", "Imagen Transformada", 0, 200, trans)

while True:
    # Leer fotograma
    _, src = cap.read()
    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) 
    # Mostrar el fotograma
    cv2.imshow("Imagen Original", gray)
    
    # Matriz para la imagen transformada
    dst = np.zeros((dimensiones[0] + 200, dimensiones[1] + 200), np.uint8)
    # Leer theta del trackbar
    theta = cv2.getTrackbarPos("theta", "Imagen Transformada")
    # Convertir grados a radianes
    theta *= math.pi / 180
    # Leer traslacion del trackbar
    t = cv2.getTrackbarPos("t", "Imagen Transformada")
    
    # Transformar cada pixel en la imagen
    for i in range(dimensiones[0]):
        for j in range(dimensiones[1]):
            # Realizar rotacion
            x = math.ceil(i * math.cos(theta) + j * math.sin(theta))
            y = math.ceil(-i * math.sin(theta) + j * math.cos(theta))
            
            # Evitar salir de los limites
            if x + t > dst.shape[0] - 1:
                x = dst.shape[0] - 1 - t
            if y + t > dst.shape[1] - 1:
                y = dst.shape[1] - 1 - t
            if x + t < 0:
                x = 0
            if y + t < 0:
                y = 0
            # Asignar las nuevas coordenadas
            dst[x + t][y + t] = gray[i][j]
    cv2.imshow("Imagen Transformada", dst)

    # Salir si se presiona la q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break




