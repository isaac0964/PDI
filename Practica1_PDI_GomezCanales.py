# Importar librerias requeridas
import cv2
import numpy as np
import math

"""
Cambios:
1. Cambiar el shape de dst de (m, n, 1) a (m, n)
2. Rellenar espacios en negro con np.where
3. Cambiar el signo de la rotacion para hacerla en sentido antihorario 
4. Hacer que la barra gris (lo que se sale de los limites de la imagen) se quede en las orillas

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
def transx(x):
    pass
def transy(x):
    pass
def scalex(x):
    pass
def scaley(x):
    pass

# Crear la figura donde se mostrara la imagen transformada y el trackbar
cv2.namedWindow("Imagen Transformada")
cv2.createTrackbar("theta", "Imagen Transformada", 0, 360, rot)
cv2.createTrackbar("tx", "Imagen Transformada", 0, 200, transx)
cv2.createTrackbar("ty", "Imagen Transformada", 0, 200, transy)
cv2.createTrackbar("sx", "Imagen Transformada", 1, 20, scalex)
cv2.createTrackbar("sy", "Imagen Transformada", 1, 20, scaley)

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

    # Leer traslacion del trackbar en x y en y
    tx = cv2.getTrackbarPos("tx", "Imagen Transformada")
    ty = cv2.getTrackbarPos("ty", "Imagen Transformada")

    # Leer la escala del trackbar en x y en y
    sx = cv2.getTrackbarPos("sx", "Imagen Transformada")
    sy = cv2.getTrackbarPos("sy", "Imagen Transformada")
    
    # Transformar cada pixel en la imagen
    for i in range(dimensiones[0]):
        for j in range(dimensiones[1]):
            # Realizar rotacion
            x = math.ceil(i * math.cos(theta) - j * math.sin(theta))
            y = math.ceil(i * math.sin(theta) + j * math.cos(theta))

            # Realizar escalamiento
            x *= sx
            y *= sy
            
            # Evitar salir de los limites
            if x + tx > dst.shape[0] - 1:
                x = dst.shape[0] - 1 - tx
            if y + ty > dst.shape[1] - 1:
                y = dst.shape[1] - 1 - ty
            if x + tx < 0:
                x = 0 - tx
            if y + ty < 0:
                y = 0 - ty

            # Asignar las nuevas coordenadas
            dst[x + tx][y + ty] = gray[i][j]

    # Rellenar espacios negros
    xs, ys = np.where(dst == 0)  # Obtener las coords de los pixeles en negro
    # Clipear los limites para que al sumar el 1 el indice sea mayor que el tamano de la img
    ys = np.clip(ys, 0, dst.shape[1] - 2)
    dst[xs, ys] = dst[xs, (ys+1)]
            
    cv2.imshow("Imagen Transformada", dst)

    # Salir si se presiona la q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break