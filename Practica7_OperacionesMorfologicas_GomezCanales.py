# Angel Isaac Gomez Canales
# Operaciones Morfologicas 12/03/2024

# Importar librerias
import cv2 as cv
import numpy as np


def obtener_ref(cap):
    """
    Esta funcion obtiene la imagen de referencia para detectar movimiento desde la webcam
    - Input: cap: webcam
    - Output: I_ref: Imagen de referencia
    """
    while True:
        _, I_ref = cap.read()  # Leer imagen de la webcam
        # Convertir a escala de grises
        I_ref = cv.cvtColor(I_ref, cv.COLOR_BGR2GRAY)
        cv.imshow("Imagen Referencia (presione q para capturar imageb)", I_ref)
        # Obtener imagen cuando se presione q
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            return I_ref
        
def detectar_mov(Ig, I_ref, kernel, it_er, it_dil):
    """
    Esta funcion decta movimiento en una imagen usando la imagen de referencia
    - Input:
        - Ig: Imagen en escala de grises sobre la que se quiere detectar el movimiento
        - I_ref: Imagen de referencia
        - kernel: kernel para dilatacion y erosion
        - it_er: numero de erosiones
        - it_dil: numero de dilataciones
    - Output: keypoints: blobs detectados
    """
    # Suavizar las imgs
    I_ref = cv.GaussianBlur(I_ref, (21, 21), -1)
    Ig = cv.GaussianBlur(Ig, (21, 21), -1)

    I_diff = np.abs(I_ref - Ig)  # Diferencia entre la imagen de referencia y la actual
    # cv.threshold(I, thresh, valor si I(x,y) > thresh, metodo de umbralizacion)
    _, I_diff_bin = cv.threshold(I_diff, 100, 255, cv.THRESH_BINARY)  # Binarizar la imagen de diferencia e invertirla
    
    # Erosionar y dilatar la imagen binaria 
    im_salida = cv.erode(I_diff_bin, kernel, iterations=it_er)
    im_salida = cv.dilate(im_salida, kernel, iterations=it_dil)

    cv.imshow("2", im_salida)

    # La imagen se invierte porque SimpleBlobDetector detecta blobs negros
    im_salida = cv.bitwise_not(im_salida)
    keypoints = detector.detect(im_salida)
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

# Obtener Imagen de referencia
I_ref = obtener_ref(cap)

# Crear Detector de manchas y sus parametros
params = cv.SimpleBlobDetector_Params()
#params.minThreshold = 20
#params.maxThreshold = 200

# Filtar por area
params.filterByArea = True
params.minArea = 500
# Filtrar por circularidad
params.filterByCircularity = False
# Filtrar por Convexidad
params.filterByConvexity = False
# Filtrar por inercia
params.filterByInertia = False

detector = cv.SimpleBlobDetector_create(params)

# Kernel de la erosion y dilatacion
kernel = np.ones((5, 5), np.uint8)

# Crear ventana y trackbars
cv.namedWindow("Deteccion")
cv.createTrackbar("it_er", "Deteccion", 1, 20, it_er)
cv.createTrackbar("it_dil", "Deteccion", 1, 20, it_dil)

# Realizar la deteccion
while True:
    # Leer imagen de la webcam
    _, I = cap.read()
    # Convertir I a escala de grises
    Ig = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

    # Mostar la referencia
    cv.imshow("Referencia", I_ref)

    # Obtener del trackbar el numero de erosiones y dilataciones
    it_er = cv.getTrackbarPos("it_er", "Deteccion")
    it_dil = cv.getTrackbarPos("it_dil", "Deteccion")

    # Realizar deteccion
    keypoints = detectar_mov(Ig, I_ref, kernel, it_er, it_dil)

    # Dibujar los keypoints sobre la imagen en la que se detecta el movimiento
    # cv.drawKeypoints(I, keypoints, img_out, color, flags)
    # img_out es la imagen de salida donde se dibujaran los keypoints
    I_mov = cv.drawKeypoints(I, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.putText(I_mov, f"Numero de objetos encontrados: {len(keypoints)}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar deteccion de movimiento
    cv.imshow("Deteccion", I_mov)

    if cv.waitKey(1) == ord("q"):
        break


