# Angel Isaac Gomez Canales
# Operaciones Morfologicas 12/03/2024

# Importar librerias
import cv2 as cv
import numpy as np

# Funcion para obtener ref
def generar_ref(cap):
    """
    Esta funcion genera una imagen de referencia para la deteccion de movimienot
    - Input: cap: webcam
    - Output: Imagen de referencia
    """
    while True:
        # Mostrar webcam
        _, I = cap.read()
        cv.imshow("Imagen (presionar q para obtener la referencia)", I)
        if cv.waitKey(1) == ord('q'):
            return I
        
def detectar_mov(Ig, I_ref, kernel, it_er, it_dil):
    """
    Esta funcion decta movimiento en una imagen usando la imagen de referencia
    - Input:
        - Ig: Imagen en escala de grises sobre la que se quiere detectar el movimiento
        - I_ref: Imagen de referencia (en escala de grises)
        - kernel: kernel para dilatacion y erosion
        - it_er: numero de erosiones
        - it_dil: numero de dilataciones
    - Output: keypoints: blobs detectados
    """
    # Suavizar las imagenes
    I_ref = cv.GaussianBlur(I_ref, (21, 21), -1)
    Ig = cv.GaussianBlur(Ig, (21, 21), -1)

    # Obtener diferencia y binarizar
    I_diff = cv.absdiff(I_ref, Ig) # Diferencia entre la imagen de referencia y la actual
    # cv.threshold(I, thresh, valor si I(x,y) > thresh, metodo de umbralizacion)
    _, I_diff_bin = cv.threshold(I_diff, 30, 255, cv.THRESH_BINARY)  # Binarizar la imagen de diferencia e invertirla

    # Erosionar y dilatar la imagen binaria 
    im_salida = cv.erode(I_diff_bin, kernel, iterations=it_er)
    im_salida = cv.dilate(im_salida, kernel, iterations=it_dil)

    cv.imshow("Segmentacion movimento", im_salida)
    # Detectar contornos de objetos en movimiento
    keypoints, _ = cv.findContours(im_salida, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return keypoints


# Funciones para los trackbars
def it_er(x):
    pass

def it_dil(x):
    pass
        
# Configurar Webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Kernel de la erosion y dilatacion
kernel = np.ones((3, 3), np.uint8)

# Obtener image de referencia y pasarla a escala de grises
I_ref = cv.cvtColor(generar_ref(cap), cv.COLOR_BGR2GRAY)
cv.destroyAllWindows()

# Crear ventana y trackbars
cv.namedWindow("Deteccion")
cv.createTrackbar("it_er", "Deteccion", 1, 20, it_er)
cv.createTrackbar("it_dil", "Deteccion", 1, 20, it_dil)

# 1. Tomar la referencia constantemente
# 2. Cambiar a HSV

# Realizar la deteccion
while True:
    _, I = cap.read()  # Imagen actual
    Ig = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    # Obtener del trackbar el numero de erosiones y dilataciones
    it_er = cv.getTrackbarPos("it_er", "Deteccion")
    it_dil = cv.getTrackbarPos("it_dil", "Deteccion")

    # Realizar deteccion
    keypoints = detectar_mov(Ig, I_ref, kernel, it_er, it_dil)

    # Dibujar los contornos en la imagen
    for c in keypoints:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(I, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(I, f"Numero de objetos en movimiento: {len(keypoints)}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar deteccion de movimiento
    cv.imshow("Deteccion", I)

    if cv.waitKey(1) == ord("q"):
        break


